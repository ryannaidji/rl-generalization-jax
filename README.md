# rl-generalization-jax

**Does a PPO agent that does well on its training task variations also do well on held-out ones?** A small, reproducible study, with PPO written from scratch in JAX and a report written like a short paper ([`report/report.md`](report/report.md)).

> **Status: no measured results yet.** The full Part A study has not been run, and Part B has not been run on Colab. No numbers are claimed here until the result files exist.

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

CartPole is small and physics variation covers four parameters only. This is not peer reviewed and is not a publication. See the report for more. There is no screenshot or recorded demo output yet.
