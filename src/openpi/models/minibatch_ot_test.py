import jax
import jax.numpy as jnp

from openpi.models import minibatch_ot
from openpi.models import pi0_config
from openpi.shared import nnx_utils


def _structured_batch(key, batch_size=16, action_horizon=4, action_dim=3):
    actions_key, noise_key, perm_key = jax.random.split(key, 3)
    actions = 10.0 * jax.random.normal(actions_key, (batch_size, action_horizon, action_dim))
    permutation = jax.random.permutation(perm_key, batch_size)
    # Noise is a permuted, slightly perturbed copy of the actions, so the OT pairing is known.
    noise = actions[permutation] + 0.01 * jax.random.normal(noise_key, actions.shape)
    return actions, noise


def test_couple_noise_to_actions_recovers_known_pairing():
    actions, noise = _structured_batch(jax.random.key(0))
    coupled = minibatch_ot.couple_noise_to_actions(noise, actions)
    # After coupling, each noise sample should sit next to its action partner.
    assert jnp.allclose(coupled, actions, atol=0.5)


def test_coupling_reduces_independent_transport_cost():
    actions, noise = _structured_batch(jax.random.key(1))
    coupled = minibatch_ot.couple_noise_to_actions(noise, actions)
    independent_cost = jnp.mean(jnp.sum(jnp.square(noise - actions), axis=(-1, -2)))
    coupled_cost = jnp.mean(jnp.sum(jnp.square(coupled - actions), axis=(-1, -2)))
    assert coupled_cost < independent_cost


def test_couple_noise_to_actions_is_jittable():
    actions, noise = _structured_batch(jax.random.key(2))
    coupled = jax.jit(minibatch_ot.couple_noise_to_actions)(noise, actions)
    assert coupled.shape == noise.shape
    assert jnp.all(jnp.isfinite(coupled))


def test_pi0_ot_coupling_flag_defaults_off():
    assert pi0_config.Pi0Config().use_ot_coupling is False


def test_pi0_dummy_model_with_ot_coupling():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy", use_ot_coupling=True)
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)
    assert jnp.all(jnp.isfinite(loss))
