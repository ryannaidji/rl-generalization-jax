"""CLI demo: train one short PPO run, print return on train and held-out physics.

Usage: python -m experiments.demo [--steps 150000] [--n-train 8]
"""
import argparse
import time

import jax

from experiments import config as C
from rlgen import contexts as ctx
from rlgen.evaluate import evaluate, greedy_policy, random_policy
from rlgen.ppo import Config, make_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--n-train", type=int, default=8)
    args = ap.parse_args()
    key = jax.random.PRNGKey(0)
    k_ctx, k_int, k_ext, k_eval = jax.random.split(key, 4)
    train_ctx = ctx.training_contexts(k_ctx, args.n_train)
    t = time.time()
    params, m = jax.jit(make_train(Config(total_steps=args.steps, lr=1e-3)))(key, train_ctx)
    jax.block_until_ready(params)
    print(f"trained {args.steps} env steps on {args.n_train} contexts in {time.time() - t:.1f}s "
          f"(backend: {jax.default_backend()})")
    print(f"final explained variance {float(m['explained_variance'][-1]):.3f}, "
          f"approx KL {float(m['approx_kl'][-1]):.4f}")
    sets = {"train contexts": train_ctx, "held-out interpolation": ctx.sample_box(k_int, 16),
            "held-out extrapolation": ctx.sample_extrapolation(k_ext, 16)}
    print(f"{'set':<26}{'PPO greedy':>12}{'random':>10}")
    for name, c in sets.items():
        ppo_r = float(evaluate(greedy_policy(params), k_eval, c, C.EVAL_EPISODES).mean())
        rand = float(evaluate(random_policy, k_eval, c, C.EVAL_EPISODES).mean())
        print(f"{name:<26}{ppo_r:>12.1f}{rand:>10.1f}")


if __name__ == "__main__":
    main()
