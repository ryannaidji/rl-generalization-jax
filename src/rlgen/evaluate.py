"""Evaluation of a fixed policy on a set of contexts, and the non-learned baselines."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import env as E
from .ppo import actor_logits


def greedy_policy(params):
    def act(key, obs):
        return jnp.argmax(actor_logits(params, obs), axis=-1)
    return act


def random_policy(key, obs):
    return jax.random.randint(key, (), 0, E.NUM_ACTIONS)


def heuristic_policy(key, obs):
    """Hand-written rule: push toward the side the pole is falling (theta + theta_dot)."""
    return (obs[..., 2] + obs[..., 3] > 0).astype(jnp.int32)


def episode_return(act, key, params: E.EnvParams):
    """Undiscounted return of one episode (at most MAX_STEPS)."""
    k_reset, k_loop = jax.random.split(key)
    obs, state = E.reset(k_reset, params)

    def body(carry, k):
        obs, state, alive = carry
        action = act(k, obs)
        obs, state, reward, term, trunc = E.step(state, action)
        ret = reward * alive
        alive = alive * (1.0 - (term | trunc).astype(jnp.float32))
        return (obs, state, alive), ret

    _, rewards = jax.lax.scan(body, (obs, state, 1.0), jax.random.split(k_loop, E.MAX_STEPS))
    return rewards.sum()


def evaluate(act, key, contexts: E.EnvParams, episodes: int = 8):
    """Per-context mean return, shape (n_contexts,). Each context gets `episodes` episodes."""
    n = jax.tree_util.tree_leaves(contexts)[0].shape[0]
    keys = jax.random.split(key, n * episodes).reshape(n, episodes, -1)

    def one_ctx(ks, ctx):
        return jax.vmap(lambda k: episode_return(act, k, ctx))(ks).mean()

    return jax.vmap(one_ctx)(keys, contexts)
