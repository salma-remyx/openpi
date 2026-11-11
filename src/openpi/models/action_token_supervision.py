"""Auxiliary action-token supervision for mixture-of-transformer VLA models (pi0 / pi0.5).

Adapted from "Should the Action Expert Train the VLM? Action-Token Supervision and Gradient
Routing in Mixture-of-Transformer VLAs" (arXiv:2611.05249). The paper disentangles the two
signals that can teach a pre-trained VLM representations useful for control:

1. coupling: gradients backpropagated from the action expert's flow-matching loss into the VLM
   (configurable via `gradient_routing` in `pi0_config.Pi0Config`), and
2. action-token supervision: training the VLM itself to autoregressively predict discrete tokens
   encoding the same target actions.

This module implements (2) for the JAX and PyTorch backends. Target actions are discretized into
`n_bins` uniform bins over [-1, 1] (the assumed post-normalization range, following the state
binning convention in `openpi.models.tokenizer`) and mapped onto a dedicated slice of the
PaliGemma vocabulary (the same `vocab_size - 1 - 128 - bin` mapping used by `BinningTokenizer`).
The supervised pass runs the VLM trunk only: the action tokens are teacher-forced after the
regular prefix and predicted causally with a weight-tied LM head. Keeping this pass separate from
the flow-matching forward pass is deliberate -- the mixture-of-transformers' shared attention
mask cannot both show the target tokens to the VLM and hide them from the action expert, and
letting the expert attend to its own targets would leak the labels into the denoiser input.
"""

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models.gemma import PALIGEMMA_VOCAB_SIZE

# Number of discretization bins per action dimension (matches the state binning convention).
N_BINS = 256
# Skip the last 128 vocabulary entries (special tokens), as in `BinningTokenizer`.
SPECIAL_TOKEN_MARGIN = 128


def _bin_boundaries(n_bins: int) -> np.ndarray:
    """Bin edges over [-1, 1], matching `np.digitize(x, bins=np.linspace(-1, 1, n_bins + 1)[:-1]) - 1`."""
    return np.linspace(-1.0, 1.0, n_bins + 1)[:-1]


def _token_ids_from_bins(bins):
    return PALIGEMMA_VOCAB_SIZE - 1 - SPECIAL_TOKEN_MARGIN - bins


def tokenize_actions(actions, *, n_bins: int = N_BINS) -> jax.Array:
    """Discretizes target actions into PaliGemma vocabulary token ids.

    Actions (assumed normalized to [-1, 1]) are binned per dimension with the same boundaries as
    `np.digitize(x, bins=np.linspace(-1, 1, n_bins + 1)[:-1]) - 1` and flattened horizon-major, so
    token t encodes `actions[..., t // action_dim, t % action_dim]`.

    Args:
        actions: float array of shape [..., action_horizon, action_dim].
        n_bins: number of bins per action dimension.

    Returns:
        int32 array of shape [..., action_horizon * action_dim] with token ids in a dedicated
        slice at the end of the PaliGemma vocabulary (excluding special tokens).
    """
    boundaries = jnp.asarray(_bin_boundaries(n_bins))
    flat = actions.reshape(*actions.shape[:-2], -1)
    bins = jnp.sum(flat[..., None] >= boundaries, axis=-1) - 1
    return _token_ids_from_bins(jnp.clip(bins, 0, n_bins - 1)).astype(jnp.int32)


def action_token_ce_loss(llm, prefix_tokens, prefix_mask, actions, *, n_bins: int = N_BINS) -> jax.Array:
    """Teacher-forced cross-entropy loss of the VLM trunk on discretized target actions.

    Args:
        llm: the mixture-of-transformers `Module` wrapped in `nnx_bridge.ToNNX` (as exposed by
            `Pi0.PaliGemma.llm`); only its first expert (the VLM trunk) is run.
        prefix_tokens: float array [..., prefix_len, emb] of embedded prefix (image/language) tokens.
        prefix_mask: bool array [..., prefix_len] marking valid prefix tokens.
        actions: float array [..., action_horizon, action_dim] of target actions.
        n_bins: number of discretization bins per action dimension.

    Returns:
        float array of shape [...] with the mean per-token cross entropy.
    """
    from openpi.models.pi0 import make_attn_mask

    token_ids = tokenize_actions(actions, n_bins=n_bins)
    num_tokens = token_ids.shape[-1]
    prefix_len = prefix_tokens.shape[-2]

    token_embeddings = llm(token_ids, method="embed")
    embedded = jnp.concatenate([prefix_tokens, token_embeddings], axis=-2)
    input_mask = jnp.concatenate([prefix_mask, jnp.ones(token_ids.shape, dtype=jnp.bool_)], axis=-1)
    # The prefix attends bidirectionally; each action token attends strictly causally, so that
    # token i is predicted from the prefix and the preceding action tokens only.
    ar_mask = jnp.concatenate([jnp.zeros_like(prefix_mask), jnp.ones(token_ids.shape, dtype=jnp.bool_)], axis=-1)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=-1) - 1

    (trunk_out, expert_out), _ = llm([embedded, None], mask=attn_mask, positions=positions)
    assert expert_out is None

    # The last valid prefix position predicts the first action token, action token position i-1
    # predicts token i.
    last_valid = jnp.sum(prefix_mask, axis=-1) - 1
    gather_index = jnp.broadcast_to(last_valid[..., None, None], (*trunk_out.shape[:-2], 1, trunk_out.shape[-1]))
    boundary = jnp.take_along_axis(trunk_out, gather_index, axis=-2)
    predictors = jnp.concatenate([boundary, trunk_out[..., prefix_len : prefix_len + num_tokens - 1, :]], axis=-2)

    log_probs = jax.nn.log_softmax(llm(predictors, method="decode"), axis=-1)
    token_ce = -jnp.take_along_axis(log_probs, token_ids[..., None], axis=-1)[..., 0]
    return jnp.mean(token_ce, axis=-1)


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
