"""Minibatch optimal-transport coupling for flow-matching training.

Implements the core mechanism of "Optimal Flow Matching: Learning Straight Trajectories in Just One Step"
(Kornilov et al., arXiv:2403.13117): instead of pairing noise and action samples independently within a
training batch, solve an optimal-transport problem over the batch and pair each noise sample with its
assigned action sequence. OT couplings straighten the learned flow's trajectories, so fewer integration
steps are needed at inference time (the inference path itself is unchanged).

The paper solves the minibatch OT problem exactly with a linear-assignment solver. Exact assignment
solvers do not run inside a jitted, sharded JAX training step, so this module substitutes entropic OT
(Sinkhorn) with argmax rounding -- a standard jittable approximation of minibatch OT. The coupling is
parameter-free and computed under `jax.lax.stop_gradient`: it only re-pairs existing samples, so the
loss I/O and the model are untouched.
"""

import jax
import jax.numpy as jnp
import jax.scipy.special


def ot_assignment(noise: jax.Array, actions: jax.Array, *, num_iters: int = 100, epsilon: float = 0.1) -> jax.Array:
    """Computes an approximate minibatch-OT assignment of noise samples to action sequences.

    Args:
      noise: Noise samples, `(*b ah ad)`.
      actions: Action sequences, `(*b ah ad)`. Leading batch dims must match `noise`.
      num_iters: Number of Sinkhorn iterations.
      epsilon: Entropic regularization strength, applied to the mean-normalized cost.

    Returns:
      Int array `pi` of shape `(*b,)` such that `noise[pi[j]]` is the OT partner of `actions[j]`.
    """
    flat_noise = jax.lax.stop_gradient(noise.reshape(-1, noise.shape[-2] * noise.shape[-1]))
    flat_actions = jax.lax.stop_gradient(actions.reshape(-1, actions.shape[-2] * actions.shape[-1]))

    cost = jnp.sum(jnp.square(flat_noise[:, None, :] - flat_actions[None, :, :]), axis=-1)
    # Normalize the cost scale so `epsilon` is meaningful across action dims and datasets.
    cost = cost / (jnp.mean(cost) + 1e-8)
    log_kernel = -cost / epsilon

    batch_size = flat_noise.shape[0]
    log_marginal = -jnp.log(batch_size)

    def sinkhorn_step(carry, _):
        log_u, log_v = carry
        log_u = log_marginal - jax.scipy.special.logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_marginal - jax.scipy.special.logsumexp(log_kernel + log_u[:, None], axis=0)
        return (log_u, log_v), None

    init = (jnp.zeros(batch_size), jnp.zeros(batch_size))
    (log_u, log_v), _ = jax.lax.scan(sinkhorn_step, init, None, length=num_iters)

    log_plan = log_u[:, None] + log_kernel + log_v[None, :]
    # Round the transport plan to a hard assignment: each action gets the noise sample carrying the
    # most planned transport mass. Collisions are possible in principle but rare for small epsilon,
    # and the transport-cost reduction does not rely on the assignment being a perfect permutation.
    return jnp.argmax(log_plan, axis=0).reshape(noise.shape[:-2])


def couple_noise_to_actions(
    noise: jax.Array, actions: jax.Array, *, num_iters: int = 100, epsilon: float = 0.1
) -> jax.Array:
    """Permutes `noise` so that sample `i` is the minibatch-OT partner of `actions[i]`."""
    pi = ot_assignment(noise, actions, num_iters=num_iters, epsilon=epsilon)
    return noise[pi]
