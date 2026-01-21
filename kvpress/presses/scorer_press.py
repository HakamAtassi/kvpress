# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.base_press import BasePress

logger = logging.getLogger(__name__)


@dataclass
class ScorerPress(BasePress):
    """
    Base class for score-based KV cache compression methods.

    This class assigns scores to key-value pairs and prune those with the lowest scores.
    Subclasses then implement the `score` method to define how importance is calculated.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    """

    compression_ratio: float = 0.0
    pruned_ages:list[int] = field(default_factory=list)

    ##########################
    # migration scheme stuff #
    ##########################

    HBM_token_buffer: int = 256 # max number of tokens you can have in the HBM
    migration_counter:int = field(default_factory=int)  # number of tokens migrated to HBF across entire run (total)
    pruned_migration_counter:int = field(default_factory=int) # number of entries that were pruned after migration


    # migration metadata
    alpha = 0.7
    gamma = 0.0  # volatility penalty weight
    age_saturation = 256 # Age at which token is "fully mature"


    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """
        Compute importance scores for each key-value pair.

        This method must be implemented by subclasses to define how the importance
        of each token position is calculated. Higher scores indicate more important
        tokens that should be kept during compression.

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer where scoring is applied.
        hidden_states : torch.Tensor
            Input embeddings with shape (batch_size, seq_len, hidden_dim).
        keys : torch.Tensor
            Key tensors with shape (batch_size, num_kv_heads, seq_len, head_dim).
        values : torch.Tensor
            Value tensors with shape (batch_size, num_kv_heads, seq_len, head_dim).
        attentions : torch.Tensor
            Attention weights with shape (batch_size, num_heads, seq_len, seq_len).
            May be None if not computed or needed by the scoring method.
        kwargs : dict
            Additional arguments from the forward pass, including cache and position info.

        Returns
        -------
        torch.Tensor
            Importance scores with shape (batch_size, num_kv_heads, seq_len).
            Higher scores indicate more important tokens. The tokens with the
            lowest scores will be pruned during compression.
        """
        raise NotImplementedError

    def migrate_random_excess(self, keys, values, ages, migrated, healtha):
        """
        Randomly migrate the excess tokens.

        For each (batch, head), if the number of tokens on HBM (migrated==0) exceeds
        self.HBM_token_buffer by k, then randomly select k unmigrated tokens and set
        their migrated flag to 1 across the full head_dim.

        Shapes
        ------
        migrated: [B, H, S, D]
        (keys/values/ages are unused here, kept for signature compatibility)
        """
        # migrated: [B, H, S, D]  (D is a replicated “bit”/flag dim)
        B, H, S, D = migrated.shape

        # [B, H, S] (all of last dim are identical, so just take one)
        migrated_tokens = migrated[..., 0]

        # True where token is currently on HBM (per your convention: 0 == on HBM)
        on_hbm_mask = (migrated_tokens == 0)                         # [B, H, S]
        on_hbm_count = on_hbm_mask.sum(dim=-1)                       # [B, H]

        # number of *excess* tokens on HBM per head
        excess_per_head = (on_hbm_count - self.HBM_token_buffer).clamp(min=0)  # [B, H]

        # sample excess tokens (random, w/out replacement) from those currently on HBM
        for b in range(B):
            for h in range(H):
                k = int(excess_per_head[b, h].item())
                if k <= 0:
                    continue

                # indices (positions along seq_len) that are on HBM
                on_idx = torch.nonzero(on_hbm_mask[b, h], as_tuple=False).squeeze(-1)  # [N]
                if on_idx.numel() == 0:
                    continue

                # safety: cannot sample more than available
                k = min(k, on_idx.numel())

                # random sample without replacement
                perm = torch.randperm(on_idx.numel(), device=on_idx.device)[:k]
                chosen = on_idx[perm]  # [k]

                # set migration bit/flag to 1 for those positions (across last dim)
                migrated[b, h, chosen, :] = 1

                # increment counters
                # (if you keep a scalar counter)
                self.migration_counter += k

        return migrated














    def migrate_oldest(self, keys, values, ages, migrated, health):
        """
        Migrate the oldest tokens first.

        For each (batch, head), if the number of tokens on HBM exceeds self.HBM_token_buffer,
        migrate the oldest `excess` unmigrated tokens by setting migrated[..., :] = 1 for those tokens.

        Shapes
        ------
        migrated: [B, H, S, D]
        ages:     [B, H, S, D] (token-level age is ages[..., 0])
        """
        B, H, S, D = migrated.shape

        migrated_tokens = migrated[..., 0]   # [B, H, S]
        token_ages = ages[..., 0]            # [B, H, S]

        # unmigrated == on HBM
        on_hbm_mask = migrated_tokens == 0   # [B, H, S]

        # per-(B,H) count of tokens currently on HBM
        on_hbm_count = on_hbm_mask.sum(dim=-1)  # [B, H]

        # how many we must migrate per (B,H)
        excess = (on_hbm_count - self.HBM_token_buffer).clamp(min=0)  # [B, H]

        if excess.max().item() == 0:
            return migrated

        # Flatten (B,H) to loop cleanly while keeping logic correct per head
        flat_mask = on_hbm_mask.view(-1, S)      # [(B*H), S]
        flat_ages = token_ages.view(-1, S)       # [(B*H), S]
        flat_excess = excess.view(-1)            # [(B*H)]

        device = migrated.device
        neg_inf = torch.finfo(torch.float32).min

        for bh in range(flat_mask.shape[0]):
            k = int(flat_excess[bh].item())
            if k <= 0:
                continue

            # Score only unmigrated tokens by age; exclude others with -inf
            age_scores = flat_ages[bh].float().masked_fill(~flat_mask[bh], neg_inf)

            # Choose the k oldest token indices
            chosen = age_scores.topk(k, largest=True).indices  # [k] on same device

            b = bh // H
            h = bh % H

            # Mark as migrated across head_dim
            migrated[b, h, chosen, :] = 1

            self.migration_counter += k

        return migrated
    



    def migrate_with_health(self, keys, values, ages, migrated, health, **kwargs):
        """
        Migrate tokens based on health.

        For each (batch, head), if the number of tokens on HBM exceeds self.HBM_token_buffer,
        migrate the `excess` healthiest *currently-on-HBM* tokens by setting migrated[..., :] = 1.

        Shapes
        ------
        migrated: [B, H, S, D]
        health:   [B, H, S, D] (token-level health is health[..., 0])
        """
        B, H, S, D = migrated.shape

        migrated_tokens = migrated[..., 0]
        token_health = health[..., 0]
        token_age = ages[..., 0]

        # on HBM == unmigrated
        on_hbm_mask = (migrated_tokens == 0)          # [B, H, S]
        on_hbm_count = on_hbm_mask.sum(dim=-1)        # [B, H]

        excess = (on_hbm_count - self.HBM_token_buffer).clamp(min=0)  # [B, H]



        flat_mask = on_hbm_mask.reshape(H, S)
        flat_health = token_health.reshape(H, S)
        flat_age = token_age.reshape(H, S)
        flat_excess = excess.reshape(H, 1)


        neg_inf = torch.finfo(torch.float32).min

        for bh in range(flat_mask.shape[0]):
            k = flat_excess[bh].item() # because its a scalar. dont ask.
            if k <= 0:
                continue

            # Consider only tokens currently on HBM; exclude others with -inf
            health_scores = flat_health[bh].float().masked_fill(~flat_mask[bh], neg_inf)
            #health_scores = health_scores.float().masked_fill(flat_age[bh] < 128, neg_inf)

            head_ages = flat_age[bh]
            #median_age = head_ages.float().median()[0]
            mean_age = head_ages.float().mean()
            
            health_scores = health_scores.float().masked_fill(head_ages < mean_age, neg_inf)

            # Select k healthiest (largest health)
            chosen = health_scores.topk(k, largest=True).indices  # [k]

            b = bh // H
            h = bh % H

            migrated[b, h, chosen, :] = 1
            self.migration_counter += k


        return migrated




    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,

        keys: torch.Tensor,
        values: torch.Tensor,
        ages: torch.Tensor,
        migrated: torch.Tensor,
        health: torch.Tensor,

        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, ...]:

        if self.compression_ratio == 0:
            return keys, values, ages

        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)


        old_health = health[..., 0] # get old healths
        new_health = scores 


        moving_avg_health = (1 - self.alpha) * old_health + self.alpha * new_health
        #moving_avg_health = old_health+new_health

        health = moving_avg_health.unsqueeze(-1).expand_as(health)





        # Get indices of KV pairs with the lowest scores
        k_len = keys.shape[2]
        n_kept = int(k_len * (1 - self.compression_ratio))

        indices = scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)


        # get ages of pruned elements...
        n_pruned = k_len - n_kept
        prune_indices = scores.topk(n_pruned, dim=-1, largest=False).indices
        prune_indices = prune_indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        pruned_ages = ages.gather(2, prune_indices)[..., :, 0]
        self.pruned_ages.extend(pruned_ages.cpu().flatten().tolist())




        # if you pruned a token after it was migrated, update the counter. 
        pruned_migrated = migrated.gather(2, prune_indices)[..., :, 0]
        self.pruned_migration_counter += pruned_migrated.sum().item()


        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        ages = ages.gather(2, indices).contiguous()
        migrated = migrated.gather(2, indices).contiguous()
        health = health.gather(2, indices).contiguous()


        # After pruning, run migration 

        #migrated = self.migrate_random_excess(keys, values, ages, migrated, health)
        migrated = self.migrate_oldest(keys, values, ages, migrated, health)
        #migrated = self.migrate_with_health(keys, values, ages, migrated, health)




        return keys, values, ages, migrated, health 





    def final_ages(self, ages: torch.Tensor):
        return ages[..., :, 0].detach().cpu().flatten().tolist()







