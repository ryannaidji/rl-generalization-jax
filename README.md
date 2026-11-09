# rl-generalization-jax

**Does a PPO agent that does well on its training task variations also do well on held-out ones?** A small, reproducible study, with PPO written from scratch in JAX and a report written like a short paper ([`report/report.md`](report/report.md)).

> **Status: Part A is measured. Part B has not been run on Colab and has no results.**

## Results (Part A)

GitHub Actions CPU runner, JAX 0.4.35, CartPole with hidden physics, 10 seeds, 500,000 steps per seed, greedy evaluation on 64 held-out contexts with 8 episodes each. Mean return with 95% bootstrap CI over seeds; maximum 500. Data: [`results/part_a/study.json`](results/part_a/study.json).

| Training contexts | Own training contexts | Held-out interpolation | Held-out extrapolation |
|---|---|---|---|
| 1 (unmodified task) | 500.0 [500.0, 500.0] | 498.2 [497.0, 499.2] | 483.0 [480.5, 485.1] |
| 8 | 498.8 [497.8, 499.5] | 498.2 [496.6, 499.3] | 482.8 [480.2, 485.3] |
| 64 | 498.8 [497.9, 499.5] | 498.8 [497.8, 499.6] | 482.4 [479.9, 484.8] |
| Random policy | n/a | 21.9 | 23.8 |
| Hand-written rule | n/a | 476.4 | 441.5 |

**Honest reading:** PPO generalized well, with about 16 to 17 points lost on extrapolation, mostly from very short poles. Training on more contexts showed no detectable benefit, but this task has a **ceiling effect**: scores saturate and even a simple rule reaches 476, so the study cannot separate strong policies.

**Debugging finding:** the first run had a gradient norm near 22 against a clip of 0.5 and very small policy steps. Scaling rewards by 0.01 raised mean training return from 373.5 to 449.7 (10 seeds, non-overlapping CIs). Removing value clipping, my other hypothesis, did not help. Details and deliberate fault-injection results are in the [report](report/report.md).

## What is here

- `src/rlgen/`: CartPole with hidden physics parameters, PPO with GAE (own MLPs and loss, `jit` and `vmap` over seeds; only Adam and gradient clipping come from optax), evaluation, bootstrap statistics.
- `experiments/`: the pre-specified design (`config.py`), study runner, figures, CLI demo.
- `colab/dpo_colab.ipynb`: Part B, DPO on SmolLM2-135M for a free Colab T4.
- `.github/workflows/`: `ci.yml` (unit tests) and `study.yml` (full study on a free CPU runner, commits JSON to `results/part_a/`).

## Run

```bash
pip install -r requirements.txt
pytest -q
PYTHONPATH=src:. python -m experiments.demo --steps 150000    # short CLI demo
PYTHONPATH=src:. python -m experiments.run_study              # full study, intended for CI
```

For Part B, open the notebook in Colab with a T4 GPU, choose Run all, and save the downloaded `dpo_results.json` as `results/part_b/dpo_results.json`.

## Data and licenses

Part A uses a simulated environment and no dataset. Part B downloads `Anthropic/hh-rlhf` (MIT) and `HuggingFaceTB/SmolLM2-135M-Instruct` (Apache-2.0) from Hugging Face at run time; neither is redistributed here.

## Limitations

CartPole is small and physics variation covers four parameters only. This is not peer reviewed and is not a publication. See the report for more. The main limitation is the ceiling effect described above. Demo output, recorded from a local CPU run with a shorter budget and different learning rate than the study, is in [`docs/demo_output.txt`](docs/demo_output.txt); there is no screenshot.
