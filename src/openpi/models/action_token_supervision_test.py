"""Tests for auxiliary action-token supervision of the VLM trunk (arXiv:2611.05249)."""

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
import torch

import openpi.models.action_token_supervision as _action_tokens
import openpi.models.gemma as _gemma
from openpi.models.gemma import PALIGEMMA_VOCAB_SIZE
import openpi.models.model as _model
import openpi.models.pi0_config as _pi0_config


def _expected_token_ids(actions: np.ndarray, n_bins: int) -> np.ndarray:
    boundaries = np.linspace(-1.0, 1.0, n_bins + 1)[:-1].astype(np.float32).astype(np.float64)
    bins = np.clip(np.digitize(actions.astype(np.float64), bins=boundaries) - 1, 0, n_bins - 1)
    return PALIGEMMA_VOCAB_SIZE - 1 - _action_tokens.SPECIAL_TOKEN_MARGIN - bins.reshape(actions.shape[0], -1)


def test_tokenize_actions_matches_numpy_digitize():
    actions = np.random.RandomState(0).uniform(-1.5, 1.5, (3, 4, 5)).astype(np.float32)
    token_ids = np.asarray(_action_tokens.tokenize_actions(jnp.asarray(actions)))
    np.testing.assert_array_equal(token_ids, _expected_token_ids(actions, _action_tokens.N_BINS))
    assert token_ids.shape == (3, 20)
    assert token_ids.min() >= PALIGEMMA_VOCAB_SIZE - 1 - _action_tokens.SPECIAL_TOKEN_MARGIN - 255
    assert token_ids.max() <= PALIGEMMA_VOCAB_SIZE - 1 - _action_tokens.SPECIAL_TOKEN_MARGIN


def test_tokenize_actions_torch_matches_numpy_digitize():
    actions = np.random.RandomState(0).uniform(-1.5, 1.5, (3, 4, 5)).astype(np.float32)
    token_ids = _action_tokens.tokenize_actions_torch(torch.from_numpy(actions)).numpy()
    np.testing.assert_array_equal(token_ids, _expected_token_ids(actions, _action_tokens.N_BINS))


def _make_llm() -> nnx_bridge.ToNNX:
    config = _gemma.get_config("dummy")
    llm = nnx_bridge.ToNNX(_gemma.Module(configs=[config, config], embed_dtype="float32"))
    llm.lazy_init(rngs=nnx.Rngs(jax.random.key(0)), method="init", use_adarms=[False, False])
    return llm


def test_action_token_ce_loss_trains_only_vlm():
    graphdef, state = nnx.split(_make_llm())
    width = _gemma.get_config("dummy").width
    prefix_tokens = jax.random.normal(jax.random.key(1), (2, 5, width))
    prefix_mask = jnp.ones((2, 5), dtype=jnp.bool_)
    actions = jax.random.uniform(jax.random.key(2), (2, 4, 3), minval=-1, maxval=1)

    def loss_fn(state):
        llm = nnx.merge(graphdef, state)
        return jnp.mean(_action_tokens.action_token_ce_loss(llm, prefix_tokens, prefix_mask, actions))

    aux_loss, grads = jax.value_and_grad(loss_fn)(state)
    assert np.isfinite(float(aux_loss))
    # The action tokens are near-uniform at initialization, so the CE should be close to
    # log(vocab_size).
    assert float(aux_loss) > 0.5 * np.log(PALIGEMMA_VOCAB_SIZE)

    vlm = expert = 0.0
    for path, variable in grads.flat_state().items():
        mass = float(jnp.sum(jnp.abs(variable.value)))
        if "_1" in "/".join(map(str, path)):
            expert += mass
        else:
            vlm += mass
    # The supervised pass runs the VLM trunk only, so the action expert is not touched.
    assert vlm > 0.0
    assert expert == 0.0


def test_pi0_compute_loss_with_action_token_supervision():
    # Both transformer streams use the tiny "dummy" variant so a full Pi0 (including the SigLIP
    # tower) fits on CPU.
    def create(*, action_token_supervision: bool):
        config = _pi0_config.Pi0Config(
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_dim=8,
            action_horizon=4,
            max_token_len=16,
            dtype="float32",
            action_token_supervision=action_token_supervision,
        )
        return config.create(jax.random.key(0))

    observation = _model.Observation(
        images={
            key: jax.random.uniform(jax.random.key(1), (2, *_model.IMAGE_RESOLUTION, 3), minval=-1, maxval=1)
            for key in _model.IMAGE_KEYS
        },
        image_masks={key: jnp.ones((2,), dtype=jnp.bool_) for key in _model.IMAGE_KEYS},
        state=jax.random.normal(jax.random.key(2), (2, 8)),
        tokenized_prompt=jax.random.randint(jax.random.key(3), (2, 16), 0, 200_000),
        tokenized_prompt_mask=jnp.ones((2, 16), dtype=jnp.bool_),
    )
    actions = jax.random.uniform(jax.random.key(4), (2, 4, 8), minval=-1, maxval=1)
    rng = jax.random.key(5)

    loss_base = create(action_token_supervision=False).compute_loss(rng, observation, actions)
    loss_aux = create(action_token_supervision=True).compute_loss(rng, observation, actions)
    # Identical parameters (same init key): the difference is exactly the auxiliary CE loss,
    # broadcast over the action horizon.
    np.testing.assert_array_equal(np.asarray(loss_aux - loss_base)[:, 1:], np.asarray(loss_aux - loss_base)[:, :-1])
    assert bool(jnp.all(jnp.isfinite(loss_aux)))
    assert bool(jnp.all(loss_aux > loss_base))
