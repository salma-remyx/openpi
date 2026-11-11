"""PyTorch port of the auxiliary action-token supervision loss (arXiv:2611.05249).

PyTorch counterpart of `openpi.models.action_token_supervision` (the JAX reference), which also
defines the backend-neutral binning constants and helpers reused here.
"""

import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models.action_token_supervision import N_BINS
from openpi.models.action_token_supervision import _bin_boundaries
from openpi.models.action_token_supervision import _token_ids_from_bins


def tokenize_actions_torch(actions: torch.Tensor, *, n_bins: int = N_BINS) -> torch.Tensor:
    """PyTorch counterpart of `tokenize_actions`; see there for the binning convention."""
    boundaries = torch.from_numpy(_bin_boundaries(n_bins)).to(device=actions.device, dtype=actions.dtype)
    flat = actions.reshape(*actions.shape[:-2], -1)
    bins = torch.bucketize(flat, boundaries, right=True) - 1
    return _token_ids_from_bins(torch.clamp(bins, 0, n_bins - 1)).to(torch.int64)


def action_token_ce_loss_torch(paligemma_with_expert, prefix_embs, prefix_pad_masks, actions, *, n_bins: int = N_BINS):
    """PyTorch counterpart of `action_token_ce_loss`.

    Args:
        paligemma_with_expert: the `PaliGemmaWithExpertModel` of a `PI0Pytorch` model; only its
            VLM trunk (`paligemma.language_model`) is run.
        prefix_embs: float tensor [batch, prefix_len, emb] of embedded prefix tokens.
        prefix_pad_masks: bool tensor [batch, prefix_len] marking valid prefix tokens.
        actions: float tensor [batch, action_horizon, action_dim] of target actions.
        n_bins: number of discretization bins per action dimension.

    Returns:
        float tensor of shape [batch] with the mean per-token cross entropy.
    """
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    token_ids = tokenize_actions_torch(actions, n_bins=n_bins)
    batch_size, num_tokens = token_ids.shape
    prefix_len = prefix_embs.shape[1]
    language_model = paligemma_with_expert.paligemma.language_model

    token_embeddings = language_model.embed_tokens(token_ids).to(prefix_embs.dtype)
    embedded = torch.cat([prefix_embs, token_embeddings], dim=1)
    pad_masks = torch.cat([prefix_pad_masks, torch.ones_like(token_ids, dtype=torch.bool)], dim=1)
    att_masks = torch.cat(
        [
            torch.zeros(prefix_len, dtype=torch.bool, device=actions.device),
            torch.ones(num_tokens, dtype=torch.bool, device=actions.device),
        ]
    )[None, :].expand(batch_size, prefix_len + num_tokens)
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1

    outputs, _ = paligemma_with_expert.forward(
        attention_mask=torch.where(att_2d_masks[:, None], 0.0, -2.3819763e38),
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[embedded, None],
        use_cache=False,
    )
    trunk_out = outputs[0]

    # The last valid prefix position predicts the first action token, action token position i-1
    # predicts token i.
    last_valid = prefix_pad_masks.sum(dim=-1) - 1
    boundary = trunk_out.gather(1, last_valid[:, None, None].expand(-1, 1, trunk_out.shape[-1]))
    predictors = torch.cat([boundary, trunk_out[:, prefix_len : prefix_len + num_tokens - 1]], dim=1)

    logits = language_model.lm_head(predictors).float()
    token_ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), token_ids.reshape(-1), reduction="none")
    return token_ce.view(batch_size, num_tokens).mean(dim=-1)
