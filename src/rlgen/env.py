"""CartPole with exposed physical parameters, written in pure JAX.

Dynamics follow the classic Barto, Sutton & Anderson (1983) task as implemented in
Gym's CartPole-v1 (Euler integration, tau=0.02, 500-step limit). The physical
parameters live in `EnvParams` so that a "context" (one variation of the task) is
just a pytree that can be vmapped over. The parameters are NOT in the observation:
the agent has to act without being told which variation it is in.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

TAU = 0.02
X_LIMIT = 2.4
THETA_LIMIT = 12.0 * 2.0 * jnp.pi / 360.0
MAX_STEPS = 500
OBS_DIM = 4
NUM_ACTIONS = 2


class EnvParams(NamedTuple):
    length: jax.Array  # half pole length (m)
    masspole: jax.Array  # kg
    masscart: jax.Array  # kg
    force_mag: jax.Array  # N
    gravity: jax.Array  # m/s^2


class EnvState(NamedTuple):
    x: jax.Array
    x_dot: jax.Array
    theta: jax.Array
    theta_dot: jax.Array
    t: jax.Array
    params: EnvParams


def nominal_params() -> EnvParams:
    """Gym CartPole-v1 defaults."""
    return EnvParams(*map(jnp.asarray, (0.5, 0.1, 1.0, 10.0, 9.8)))


def get_obs(state: EnvState) -> jax.Array:
    return jnp.stack([state.x, state.x_dot, state.theta, state.theta_dot], axis=-1)


def reset(key: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:
    s = jax.random.uniform(key, (4,), minval=-0.05, maxval=0.05)
    state = EnvState(s[0], s[1], s[2], s[3], jnp.asarray(0), params)
    return get_obs(state), state


def step(state: EnvState, action: jax.Array):
    """One Euler step without auto-reset.

    Returns (obs, new_state, reward, terminated, truncated). `terminated` means the pole
    fell or the cart left the track; `truncated` means only the time limit was hit.
    """
    p = state.params
    total_mass = p.masspole + p.masscart
    polemass_length = p.masspole * p.length
    force = jnp.where(action == 1, p.force_mag, -p.force_mag)
    cos, sin = jnp.cos(state.theta), jnp.sin(state.theta)
    temp = (force + polemass_length * state.theta_dot**2 * sin) / total_mass
    theta_acc = (p.gravity * sin - cos * temp) / (
        p.length * (4.0 / 3.0 - p.masspole * cos**2 / total_mass)
    )
    x_acc = temp - polemass_length * theta_acc * cos / total_mass

    new = EnvState(
        x=state.x + TAU * state.x_dot,
        x_dot=state.x_dot + TAU * x_acc,
        theta=state.theta + TAU * state.theta_dot,
        theta_dot=state.theta_dot + TAU * theta_acc,
        t=state.t + 1,
        params=p,
    )
    terminated = (jnp.abs(new.x) > X_LIMIT) | (jnp.abs(new.theta) > THETA_LIMIT)
    truncated = (new.t >= MAX_STEPS) & ~terminated
    return get_obs(new), new, jnp.asarray(1.0), terminated, truncated


def auto_reset_step(key: jax.Array, state: EnvState, action: jax.Array, contexts: EnvParams):
    """Step, and on episode end draw a new context from `contexts` and reset.

    The returned `final_obs` is the observation the episode actually ended on. It differs
    from `obs` only when an episode ended, and is what a truncation bootstrap must use.
    """
    obs, new, reward, terminated, truncated = step(state, action)
    done = terminated | truncated
    k_ctx, k_reset = jax.random.split(key)
    n = jax.tree_util.tree_leaves(contexts)[0].shape[0]
    idx = jax.random.randint(k_ctx, (), 0, n)
    ctx = jax.tree_util.tree_map(lambda a: a[idx], contexts)
    reset_obs, reset_state = reset(k_reset, ctx)
    next_state = jax.tree_util.tree_map(lambda r, n_: jnp.where(done, r, n_), reset_state, new)
    next_obs = jnp.where(done, reset_obs, obs)
    return next_obs, next_state, reward, terminated, truncated, obs
