"""Tests for the `gradient_routing` knob: expert->VLM coupling vs. insulation (arXiv:2611.05249).

`full` (the default) propagates the action expert's flow-matching loss into the VLM (PaliGemma)
trunk; `stop_vlm` insulates the trunk. Insulation must leave the forward pass unchanged and yield
exactly zero gradients for every VLM parameter while the action expert keeps training. The PyTorch
counterpart is `openpi.models_pytorch.gemma_pytorch_test`.
"""

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.gemma as _gemma
import openpi.models.model as _model
import openpi.models.pi0_config as _pi0_config


def _make_llm(gradient_routing: str = "full") -> nnx_bridge.ToNNX:
    config = _gemma.get_config("dummy")
    llm = nnx_bridge.ToNNX(
        _gemma.Module(configs=[config, config], embed_dtype="float32", gradient_routing=gradient_routing)
    )
    llm.lazy_init(rngs=nnx.Rngs(jax.random.key(0)), method="init", use_adarms=[False, False])
    return llm


def _expert_inputs():
    width = _gemma.get_config("dummy").width
    prefix = jax.random.normal(jax.random.key(1), (2, 5, width))
    suffix = jax.random.normal(jax.random.key(2), (2, 3, width))
    mask = jnp.ones((2, 8, 8), dtype=jnp.bool_)
    positions = jnp.arange(8)[None, :].repeat(2, axis=0)
    return prefix, suffix, mask, positions


def _expert_loss(llm):
    prefix, suffix, mask, positions = _expert_inputs()
    (_, suffix_out), _ = llm([prefix, suffix], positions=positions, mask=mask)
    return jnp.mean(jnp.square(suffix_out))


def _grad_masses(gradient_routing: str) -> tuple[float, float]:
    """Runs an expert-only loss through the mixture-of-transformers; returns (vlm, expert) masses."""
    graphdef, state = nnx.split(_make_llm(gradient_routing))

    def loss_fn(state):
        return _expert_loss(nnx.merge(graphdef, state))

    vlm = expert = 0.0
    for path, variable in jax.grad(loss_fn)(state).flat_state().items():
        mass = float(jnp.sum(jnp.abs(variable.value)))
        if "_1" in "/".join(map(str, path)):
            expert += mass
        else:
            vlm += mass
    return vlm, expert


def test_coupling_updates_vlm():
    vlm, expert = _grad_masses("full")
    assert vlm > 0.0
    assert expert > 0.0


def test_stop_vlm_insulates_vlm():
    vlm, expert = _grad_masses("stop_vlm")
    assert vlm == 0.0
    assert expert > 0.0


def test_stop_vlm_leaves_mixture_forward_unchanged():
    np.testing.assert_array_equal(
        np.asarray(_expert_loss(_make_llm("full"))), np.asarray(_expert_loss(_make_llm("stop_vlm")))
    )


def test_pi0_config_defaults_and_validation():
    assert _pi0_config.Pi0Config().gradient_routing == "full"
    assert _pi0_config.Pi0Config(gradient_routing="stop_vlm").gradient_routing == "stop_vlm"
    with pytest.raises(AssertionError):
        _pi0_config.Pi0Config(gradient_routing="bogus")


def _dummy_pi0(gradient_routing: str) -> _pi0_config.Pi0Config:
    # Both transformer streams use the tiny "dummy" variant so a full Pi0 (including the SigLIP
    # tower) fits on CPU; only the forward/gradient behavior under test matters.
    return _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
        dtype="float32",
        gradient_routing=gradient_routing,
    )


def _dummy_batch() -> tuple[_model.Observation, jax.Array]:
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
    return observation, actions


def test_pi0_compute_loss_gradient_routing():
    observation, actions = _dummy_batch()
    rng = jax.random.key(5)

    # Insulation changes gradients only: with identical parameters (same init key) the losses of
    # both routings must agree bit-for-bit.
    loss_full = _dummy_pi0("full").create(jax.random.key(0)).compute_loss(rng, observation, actions)
    loss_stop = _dummy_pi0("stop_vlm").create(jax.random.key(0)).compute_loss(rng, observation, actions)
    np.testing.assert_array_equal(np.asarray(loss_full), np.asarray(loss_stop))

    graphdef, state = nnx.split(_dummy_pi0("stop_vlm").create(jax.random.key(0)))

    def loss_fn(state):
        return jnp.mean(nnx.merge(graphdef, state).compute_loss(rng, observation, actions))

    vlm = expert = 0.0
    for path, variable in jax.grad(loss_fn)(state).flat_state().items():
        name = "/".join(map(str, path))
        mass = float(jnp.sum(jnp.abs(variable.value)))
        if ("llm" in name and "_1" not in name) or "img" in name:
            vlm += mass
        elif "llm" in name:
            expert += mass
    assert vlm == 0.0
    assert expert > 0.0
