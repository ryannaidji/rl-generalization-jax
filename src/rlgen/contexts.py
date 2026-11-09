"""Train / held-out task variations. Ranges are fixed before any result is seen."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .env import EnvParams, nominal_params

# (low, high) of the training box. Gravity is held at the nominal value.
BOX = {
    "length": (0.25, 0.75),
    "masspole": (0.05, 0.20),
    "masscart": (0.7, 1.3),
    "force_mag": (7.0, 13.0),
}
# Outside-the-box bands, (low band, high band) per varied parameter.
OUTER = {
    "length": ((0.08, 0.25), (0.75, 1.5)),
    "masspole": ((0.02, 0.05), (0.20, 0.5)),
    "masscart": ((0.3, 0.7), (1.3, 2.5)),
    "force_mag": ((3.0, 7.0), (13.0, 25.0)),
}
KEYS = tuple(BOX)


def _box_values(key: jax.Array, n: int) -> jax.Array:
    lows = jnp.array([BOX[k][0] for k in KEYS])
    highs = jnp.array([BOX[k][1] for k in KEYS])
    return jax.random.uniform(key, (n, len(KEYS)), minval=lows, maxval=highs)


def _to_params(values: jax.Array) -> EnvParams:
    g = nominal_params().gravity
    return EnvParams(
        length=values[:, 0], masspole=values[:, 1], masscart=values[:, 2],
        force_mag=values[:, 3], gravity=jnp.full(values.shape[0], g),
    )


def sample_box(key: jax.Array, n: int) -> EnvParams:
    """n contexts uniformly inside the training box."""
    return _to_params(_box_values(key, n))


def sample_extrapolation(key: jax.Array, n: int) -> EnvParams:
    """n contexts with exactly one parameter outside the box, the others inside."""
    k_box, k_dim, k_side, k_u = jax.random.split(key, 4)
    values = _box_values(k_box, n)
    dims = jax.random.randint(k_dim, (n,), 0, len(KEYS))
    sides = jax.random.randint(k_side, (n,), 0, 2)
    us = jax.random.uniform(k_u, (n,))
    bands = jnp.array([[OUTER[k][0], OUTER[k][1]] for k in KEYS])  # (dim, side, lo/hi)
    lo, hi = bands[dims, sides, 0], bands[dims, sides, 1]
    return _to_params(values.at[jnp.arange(n), dims].set(lo + us * (hi - lo)))


def nominal_contexts(n: int = 1) -> EnvParams:
    return jax.tree_util.tree_map(lambda a: jnp.full((n,), a), nominal_params())


def training_contexts(key: jax.Array, n: int) -> EnvParams:
    """n == 1 means the unmodified task; n > 1 means n random draws from the box."""
    return nominal_contexts(1) if n == 1 else sample_box(key, n)


def sweep(name: str, values) -> EnvParams:
    """Nominal task with one parameter swept over `values`."""
    values = jnp.asarray(values, dtype=jnp.float32)
    base = nominal_contexts(values.shape[0])
    return base._replace(**{name: values})
