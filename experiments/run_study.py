"""Run the study: train several seeds for each condition, evaluate, write JSON.

Usage: python -m experiments.run_study [--seeds N] [--total-steps N] [--out DIR] [--skip-faults]
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments import config as C
from rlgen import contexts as ctx
from rlgen.evaluate import evaluate, greedy_policy, heuristic_policy, random_policy
from rlgen.ppo import make_train
from rlgen.stats import bootstrap_ci

DIAG_KEYS = ("train_return", "explained_variance", "approx_kl", "last_epoch_approx_kl",
             "clip_frac", "grad_norm", "max_grad_norm_seen", "v_loss", "pg_loss", "entropy")


def train_condition(cfg, n_train: int, seeds: int):
    """Train `seeds` runs in parallel with vmap. Returns (params, metrics, train contexts)."""
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(seeds))
    ctx_keys = jax.vmap(lambda k: jax.random.fold_in(k, 7))(keys)
    train_ctx = jax.vmap(lambda k: ctx.training_contexts(k, n_train))(ctx_keys)
    params, metrics = jax.jit(jax.vmap(make_train(cfg)))(keys, train_ctx)
    return params, metrics, train_ctx


def eval_params(params, key, contexts, episodes, per_seed_contexts=False):
    """Mean return over contexts per seed, shape (seeds,)."""
    def one(p, k, c):
        return evaluate(greedy_policy(p), k, c, episodes).mean()
    seeds = jax.tree_util.tree_leaves(params)[0].shape[0]
    keys = jax.random.split(key, seeds)
    return jax.jit(jax.vmap(one, in_axes=(0, 0, 0 if per_seed_contexts else None)))(
        params, keys, contexts)


def eval_sweep(params, key, contexts, episodes):
    """Per-seed, per-context returns, shape (seeds, n_contexts)."""
    def one(p, k):
        return evaluate(greedy_policy(p), k, contexts, episodes)
    seeds = jax.tree_util.tree_leaves(params)[0].shape[0]
    return jax.jit(jax.vmap(one))(params, jax.random.split(key, seeds))


def curves(metrics):
    """Per-update mean across seeds for each diagnostic, as plain lists."""
    out = {}
    for k in DIAG_KEYS:
        out[k] = [None if np.isnan(x) else round(float(x), 5)
                  for x in np.nanmean(np.asarray(metrics[k]), axis=0)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=C.SEEDS)
    ap.add_argument("--total-steps", type=int, default=C.BASE.total_steps)
    ap.add_argument("--out", default="results/part_a")
    ap.add_argument("--skip-faults", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = C.BASE._replace(total_steps=args.total_steps)
    t0 = time.time()

    k_int, k_ext, k_eval = jax.random.split(jax.random.PRNGKey(C.HELD_OUT_SEED), 3)
    held = {
        "interpolation": ctx.sample_box(k_int, C.HELD_OUT_CONTEXTS),
        "extrapolation": ctx.sample_extrapolation(k_ext, C.HELD_OUT_CONTEXTS),
    }
    sweeps = {"length": C.LENGTH_SWEEP, "force_mag": C.FORCE_SWEEP}

    def baseline(act):
        return {name: {"mean": float(evaluate(act, k_eval, c, C.EVAL_EPISODES).mean())}
                for name, c in held.items()}

    study = {
        "conditions": {},
        "baselines": {"random": baseline(random_policy), "heuristic": baseline(heuristic_policy)},
        "design": {
            "seeds": args.seeds, "total_env_steps": args.total_steps,
            "held_out_contexts": C.HELD_OUT_CONTEXTS, "eval_episodes_per_context": C.EVAL_EPISODES,
            "eval_policy": "greedy",
            "ppo": {k: v for k, v in base._asdict().items() if k != "faults"},
            "box": ctx.BOX, "outer": ctx.OUTER,
            "ci": "95% percentile bootstrap over seeds, 10000 resamples",
        },
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "jax_backend": jax.default_backend(), "jax_version": jax.__version__},
    }
    diagnostics = {}

    for n_train in C.N_TRAIN_CONDITIONS:
        t = time.time()
        params, metrics, train_ctx = train_condition(base, n_train, args.seeds)
        jax.block_until_ready(params)
        train_s = time.time() - t
        ret_train = eval_params(params, k_eval, train_ctx, C.EVAL_EPISODES, per_seed_contexts=True)
        row = {"train_contexts": bootstrap_ci(ret_train),
               "per_seed": {"train": np.asarray(ret_train).tolist()},
               "train_seconds_all_seeds": round(train_s, 1)}
        for name, c in held.items():
            ret = eval_params(params, k_eval, c, C.EVAL_EPISODES)
            row[name] = bootstrap_ci(ret)
            row[f"gap_train_minus_{name}"] = bootstrap_ci(np.asarray(ret_train) - np.asarray(ret))
            row["per_seed"][name] = np.asarray(ret).tolist()
        row["sweeps"] = {}
        for name, values in sweeps.items():
            per = np.asarray(eval_sweep(params, k_eval, ctx.sweep(name, values), C.EVAL_EPISODES))
            row["sweeps"][name] = {"values": list(values), "mean": per.mean(axis=0).tolist(),
                                   "ci": [bootstrap_ci(per[:, i]) for i in range(per.shape[1])]}
        study["conditions"][str(n_train)] = row
        diagnostics[f"n_train={n_train}"] = curves(metrics)
        print(f"n_train={n_train} train={row['train_contexts']['mean']:.1f} "
              f"interp={row['interpolation']['mean']:.1f} "
              f"extrap={row['extrapolation']['mean']:.1f} ({train_s:.0f}s)", flush=True)

    study["runtime_seconds"] = round(time.time() - t0, 1)
    (out / "study.json").write_text(json.dumps(study, indent=1))
    (out / "diagnostics.json").write_text(json.dumps(diagnostics))

    if not args.skip_faults:
        faults = {}
        for name, cfg in C.FAULTS.items():
            cfg = cfg._replace(total_steps=args.total_steps)
            params, metrics, _ = train_condition(cfg, C.FAULT_N_TRAIN, args.seeds)
            held_ret = eval_params(params, k_eval, held["interpolation"], C.EVAL_EPISODES)
            ev, kl = np.asarray(metrics["explained_variance"]), np.asarray(metrics["approx_kl"])
            gn = np.asarray(metrics["grad_norm"])
            last = max(ev.shape[1] // 4, 1)  # final quarter of the updates
            faults[name] = {
                "held_out_interpolation": bootstrap_ci(held_ret),
                "final_quarter_explained_variance": float(np.nanmean(ev[:, -last:])),
                "final_quarter_approx_kl": float(np.nanmean(kl[:, -last:])),
                "final_quarter_grad_norm": float(np.nanmean(gn[:, -last:])),
                "max_approx_kl": float(np.nanmax(kl)),
                "per_seed": np.asarray(held_ret).tolist(),
            }
            print(name, round(faults[name]["held_out_interpolation"]["mean"], 1), flush=True)
            diagnostics[f"fault={name}"] = curves(metrics)
        (out / "faults.json").write_text(json.dumps(faults, indent=1))
        (out / "diagnostics.json").write_text(json.dumps(diagnostics))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
