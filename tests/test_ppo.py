import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rlgen import contexts as ctx, ppo
from rlgen.evaluate import evaluate, greedy_policy, heuristic_policy, random_policy
from rlgen.stats import bootstrap_ci


def naive_gae(r, v, d, last, gamma, lam):
    """Plain-loop reference implementation of GAE."""
    adv = np.zeros(len(r))
    next_adv, next_v = 0.0, last
    for t in reversed(range(len(r))):
        m = 1.0 - d[t]
        delta = r[t] + gamma * m * next_v - v[t]
        next_adv = delta + gamma * lam * m * next_adv
        adv[t], next_v = next_adv, v[t]
    return adv


def test_gae_matches_reference_loop():
    rng = np.random.default_rng(0)
    r, v = rng.normal(size=50), rng.normal(size=50)
    d = (rng.random(50) < 0.1).astype(float)
    adv, tgt = ppo.compute_gae(jnp.array(r), jnp.array(v), jnp.array(d), jnp.array(0.7), 0.99, 0.95)
    np.testing.assert_allclose(adv, naive_gae(r, v, d, 0.7, 0.99, 0.95), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(tgt, np.asarray(adv) + v, atol=1e-5)


def test_gae_lambda_zero_is_td_error():
    r, v = jnp.array([1.0, 2.0]), jnp.array([0.5, 0.25])
    adv, _ = ppo.compute_gae(r, v, jnp.zeros(2), jnp.array(1.0), 0.9, 0.0)
    np.testing.assert_allclose(adv, [1.0 + 0.9 * 0.25 - 0.5, 2.0 + 0.9 - 0.25], atol=1e-6)


def test_gae_lambda_one_is_discounted_return_minus_value():
    r, v = jnp.array([1.0, 1.0, 1.0]), jnp.array([0.2, 0.3, 0.4])
    adv, _ = ppo.compute_gae(r, v, jnp.zeros(3), jnp.array(0.0), 0.5, 1.0)
    np.testing.assert_allclose(adv, [1.75 - 0.2, 1.5 - 0.3, 1.0 - 0.4], atol=1e-6)


def test_gae_does_not_leak_across_done():
    r, v = jnp.array([1.0, 100.0]), jnp.zeros(2)
    adv, _ = ppo.compute_gae(r, v, jnp.array([1.0, 0.0]), jnp.array(0.0), 0.9, 0.95)
    assert float(adv[0]) == pytest.approx(1.0)


def test_loss_is_zero_when_policy_unchanged():
    cfg = ppo.Config()
    params = ppo.init_params(jax.random.PRNGKey(0), cfg)
    obs = jax.random.normal(jax.random.PRNGKey(1), (32, 4))
    actions = jax.random.randint(jax.random.PRNGKey(2), (32,), 0, 2)
    logp = ppo.categorical_log_prob(ppo.actor_logits(params, obs), actions)
    values = ppo.critic_value(params, obs)
    adv = jax.random.normal(jax.random.PRNGKey(3), (32,))
    batch = dict(obs=obs, actions=actions, log_probs=logp, values=values, advantages=adv, targets=values)
    _, aux = ppo.ppo_loss(params, batch, cfg)
    assert float(aux["approx_kl"]) == pytest.approx(0.0, abs=1e-6)
    assert float(aux["v_loss"]) == pytest.approx(0.0, abs=1e-6)
    assert float(aux["pg_loss"]) == pytest.approx(0.0, abs=1e-5)  # normalized advantages


def test_policy_clip_bounds_the_gain():
    cfg = ppo.Config()._replace(faults=ppo.Faults(no_adv_norm=True))
    params = ppo.init_params(jax.random.PRNGKey(0), cfg)
    obs = jnp.zeros((1, 4))
    actions = jnp.array([0])
    base = ppo.categorical_log_prob(ppo.actor_logits(params, obs), actions)
    batch = dict(obs=obs, actions=actions, log_probs=base - 2.0, values=jnp.zeros(1),
                 advantages=jnp.array([1.0]), targets=jnp.zeros(1))
    _, aux = ppo.ppo_loss(params, batch, cfg)  # ratio = e^2, clipped to 1 + eps
    assert float(aux["pg_loss"]) == pytest.approx(-(1 + cfg.clip_eps), abs=1e-5)


def test_value_clip_blocks_large_value_moves():
    cfg = ppo.Config()
    params = ppo.init_params(jax.random.PRNGKey(0), cfg)
    obs = jnp.zeros((2, 4))
    v = ppo.critic_value(params, obs)
    batch = dict(obs=obs, actions=jnp.zeros(2, int), log_probs=jnp.zeros(2), values=v - 5.0,
                 advantages=jnp.zeros(2), targets=v + 100.0)
    grad = lambda c: jax.grad(lambda p: ppo.ppo_loss(p, batch, c)[0])(params)["critic"][-1]["b"]
    assert float(jnp.abs(grad(cfg)).sum()) < float(jnp.abs(grad(cfg._replace(clip_value=False))).sum())


def test_explained_variance():
    t = jnp.arange(10.0)
    assert float(ppo.explained_variance(t, t)) == pytest.approx(1.0)
    assert float(ppo.explained_variance(t, jnp.full(10, 3.0))) == pytest.approx(0.0, abs=1e-6)


def test_training_improves_on_random_smoke():
    cfg = ppo.Config(total_steps=60_000, lr=1e-3)
    params, metrics = jax.jit(ppo.make_train(cfg))(jax.random.PRNGKey(0), ctx.nominal_contexts(1))
    c = ctx.nominal_contexts(4)
    trained = float(evaluate(greedy_policy(params), jax.random.PRNGKey(1), c, 4).mean())
    rand = float(evaluate(random_policy, jax.random.PRNGKey(1), c, 4).mean())
    assert trained > 2 * rand, (trained, rand)
    assert np.isfinite(np.asarray(metrics["v_loss"])).all()


def test_baselines_ordered():
    c = ctx.nominal_contexts(8)
    h = float(evaluate(heuristic_policy, jax.random.PRNGKey(0), c, 8).mean())
    r = float(evaluate(random_policy, jax.random.PRNGKey(0), c, 8).mean())
    assert h > r


def test_bootstrap_ci_contains_mean():
    r = bootstrap_ci([1.0, 2.0, 3.0, 4.0])
    assert r["ci_low"] <= r["mean"] <= r["ci_high"] and r["n"] == 4
