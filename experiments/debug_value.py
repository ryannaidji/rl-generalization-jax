"""Follow-up to a problem seen in the first study: low explained variance, tiny approx KL.

Hypothesis: value clipping at +-0.2 is meaningless for unnormalized returns of about 100,
and the large critic gradient dominates the joint norm clip, starving the actor.
Exploratory: added after seeing results. Writes results/part_a/debug_value.json.
"""
import json
from pathlib import Path

import jax
import numpy as np

from experiments import config as C
from experiments.run_study import eval_params, train_condition
from rlgen import contexts as ctx
from rlgen.stats import bootstrap_ci

VARIANTS = {
    "as_in_study": C.BASE,
    "no_value_clip": C.BASE._replace(clip_value=False),
    "reward_scale_0.01": C.BASE._replace(reward_scale=0.01),
    "no_value_clip_reward_scale_0.01": C.BASE._replace(clip_value=False, reward_scale=0.01),
}


def window(a, lo, hi):
    n = a.shape[1]
    return float(np.nanmean(a[:, int(n * lo):int(n * hi)]))


def main():
    k_int, _, k_eval = jax.random.split(jax.random.PRNGKey(C.HELD_OUT_SEED), 3)
    held = ctx.sample_box(k_int, C.HELD_OUT_CONTEXTS)
    out = {}
    for name, cfg in VARIANTS.items():
        params, m, _ = train_condition(cfg, C.FAULT_N_TRAIN, C.SEEDS)
        ret = eval_params(params, k_eval, held, C.EVAL_EPISODES)
        tr = np.asarray(m["train_return"])
        out[name] = {
            "held_out_interpolation": bootstrap_ci(ret),
            "mean_training_return_first_quarter": bootstrap_ci(np.nanmean(tr[:, : tr.shape[1] // 4], axis=1)),
            "mean_training_return_all": bootstrap_ci(np.nanmean(tr, axis=1)),
            "explained_variance_middle_half": window(np.asarray(m["explained_variance"]), 0.25, 0.75),
            "explained_variance_final_quarter": window(np.asarray(m["explained_variance"]), 0.75, 1.0),
            "approx_kl_first_quarter": window(np.asarray(m["approx_kl"]), 0.0, 0.25),
            "approx_kl_final_quarter": window(np.asarray(m["approx_kl"]), 0.75, 1.0),
            "approx_kl_last_update": float(np.nanmean(np.asarray(m["approx_kl"])[:, -1])),
            "grad_norm_mean": window(np.asarray(m["grad_norm"]), 0.0, 1.0),
        }
        print(name, json.dumps(out[name]["held_out_interpolation"]), flush=True)
    Path("results/part_a/debug_value.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
