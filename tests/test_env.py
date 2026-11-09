import jax
import jax.numpy as jnp
import numpy as np

from rlgen import contexts as ctx, env as E


def test_nominal_matches_gym_first_step():
    # Hand-computed from the CartPole-v1 equations for theta=0, force=+10.
    p = E.nominal_params()
    s = E.EnvState(*[jnp.asarray(0.0)] * 4, jnp.asarray(0), p)
    obs, _, r, term, trunc = E.step(s, jnp.asarray(1))
    temp = 10.0 / 1.1
    theta_acc = -temp / (0.5 * (4 / 3 - 0.1 / 1.1))
    x_acc = temp - 0.05 * theta_acc / 1.1
    np.testing.assert_allclose(obs, [0.0, 0.02 * x_acc, 0.0, 0.02 * theta_acc], rtol=1e-5)
    assert r == 1.0 and not term and not trunc


def _state(theta, t, p):
    return E.EnvState(jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(theta), jnp.asarray(0.0),
                      jnp.asarray(t), p)


def test_terminates_when_pole_falls():
    *_, term, trunc = E.step(_state(0.3, 0, E.nominal_params()), jnp.asarray(0))
    assert term and not trunc


def test_truncates_only_at_time_limit():
    *_, term, trunc = E.step(_state(0.0, E.MAX_STEPS - 1, E.nominal_params()), jnp.asarray(0))
    assert trunc and not term


def test_auto_reset_returns_final_obs_and_new_episode():
    p = ctx.nominal_contexts(1)
    single = jax.tree_util.tree_map(lambda a: a[0], p)
    obs, nxt, _, term, _, final = E.auto_reset_step(
        jax.random.PRNGKey(0), _state(0.3, 7, single), jnp.asarray(0), p)
    assert term and nxt.t == 0
    assert abs(float(final[2])) > 0.2 and abs(float(obs[2])) <= 0.05


def test_extrapolation_exactly_one_param_outside_box():
    c = ctx.sample_extrapolation(jax.random.PRNGKey(1), 200)
    outside = np.zeros(200, dtype=int)
    for k in ctx.KEYS:
        v, (lo, hi) = np.asarray(getattr(c, k)), ctx.BOX[k]
        outside += (v < lo) | (v > hi)
    assert (outside <= 1).all() and (outside == 1).mean() > 0.95


def test_box_samples_inside_box():
    c = ctx.sample_box(jax.random.PRNGKey(2), 100)
    for k in ctx.KEYS:
        v, (lo, hi) = np.asarray(getattr(c, k)), ctx.BOX[k]
        assert (v >= lo).all() and (v <= hi).all()
