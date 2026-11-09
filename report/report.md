# Does PPO generalize across held-out physics? A small study in JAX

Not peer reviewed, not a publication. Small environment, small scale.

> **STATUS: NO RESULTS YET.**
> **Part A:** code and tests are complete; the full study has not been run, so this report has no Part A numbers.
> **Part B: NOT RUN. Part B has no results until `colab/dpo_colab.ipynb` is run on Colab.**

## 1. Question
If an agent performs well on the task variations it trained on, does it also perform well on variations it never saw? And does training on more distinct variations help?

## 2. Part A setup
**Environment.** CartPole-v1 dynamics in JAX (`src/rlgen/env.py`). A context is a half pole length, pole mass, cart mass and force magnitude, hidden from the agent.

**Conditions.** 1 context (the unmodified task), or 8 or 64 contexts drawn from the training box. Ten seeds per condition, each drawing its own training contexts.

| Parameter | Training box | Outside bands (extrapolation) |
|---|---|---|
| Half pole length (m) | 0.25 to 0.75 | 0.08 to 0.25, 0.75 to 1.5 |
| Pole mass (kg) | 0.05 to 0.20 | 0.02 to 0.05, 0.20 to 0.5 |
| Cart mass (kg) | 0.7 to 1.3 | 0.3 to 0.7, 1.3 to 2.5 |
| Force (N) | 7 to 13 | 3 to 7, 13 to 25 |

**Held-out sets**, the same for every seed and condition, 64 contexts each: *interpolation* (new draws from the box) and *extrapolation* (exactly one parameter outside the box). The 1-context condition has never seen most of the box, so "interpolation" is a stretch for it.

**Evaluation.** Greedy policy, 8 episodes per context. Means and 95% percentile-bootstrap intervals over seeds. The train minus held-out gap is computed per seed, then bootstrapped. Baselines: uniform random policy and a hand-written rule (push toward `theta + theta_dot`). The design is fixed in `experiments/config.py` and `src/rlgen/contexts.py` before the first full run; later changes must be recorded here.

## 3. Math
**PPO objective.** With $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{old}}(a_t|s_t)$ and advantage estimate $\hat A_t$:

$$L^{CLIP}(\theta) = \mathbb{E}_t\big[\min\big(r_t(\theta)\hat A_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t\big)\big]$$

The loss minimized is $-L^{CLIP} + c_v L^{V} - c_e H[\pi_\theta]$, with $L^V$ the clipped squared value error and $H$ the policy entropy.

**GAE.** With $\delta_t = r_t + \gamma(1-d_t)V(s_{t+1}) - V(s_t)$:

$$\hat A_t = \sum_{l\ge 0}(\gamma\lambda)^l \delta_{t+l}, \quad\text{computed as}\quad \hat A_t = \delta_t + \gamma\lambda(1-d_t)\hat A_{t+1}.$$

$\lambda=0$ gives the one-step TD error and $\lambda=1$ the discounted return minus the value; both are unit tests. Value targets are $\hat A_t + V(s_t)$.

**Time limits.** The 500-step limit is not a terminal state, so on truncation the reward is augmented with $\gamma V(s_{final})$.

**Diagnostics per update.** Explained variance $1 - \mathrm{Var}[R-V]/\mathrm{Var}[R]$, approximate KL $\mathbb{E}[(r-1) - \log r]$, clip fraction, gradient global norm before clipping.

## 4. Part A results
**Not run yet.** The study writes `results/part_a/study.json`, `faults.json`, `diagnostics.json` and `report/figures/sweeps.png`.

## 5. Debugging the training process
**Nothing to report yet.** This section will only describe problems observed in real runs, with diagnostics and fixes. The fault-injection experiment (`FAULTS` in `experiments/config.py`) re-introduces four mistakes on purpose and will be labelled as injected.

## 6. Part B: DPO on a small language model
**NOT RUN. There are no Part B results.** `colab/dpo_colab.ipynb` trains `HuggingFaceTB/SmolLM2-135M-Instruct` with a hand-written DPO loss on 2000 pairs from `Anthropic/hh-rlhf` (`helpful-base`), and measures preference accuracy on 500 training pairs and 500 held-out pairs across 3 seeds, against the untrained reference model and a "longer response wins" baseline. Output belongs at `results/part_b/dpo_results.json`.

## 7. Limitations
- CartPole is small and low-dimensional; conclusions may not carry over.
- Physics variation only, along four parameters.
- Ten seeds give wide intervals, and a percentile bootstrap at this sample size is somewhat too narrow.
- One hyperparameter setting, not tuned per condition.
