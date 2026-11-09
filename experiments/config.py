"""Pre-specified study design. Committed before the first full study run."""
from rlgen.ppo import Config, Faults

SEEDS = 10
N_TRAIN_CONDITIONS = (1, 8, 64)  # number of distinct training contexts
HELD_OUT_CONTEXTS = 64
EVAL_EPISODES = 8
HELD_OUT_SEED = 12345  # held-out sets are the same for every condition and seed
BASE = Config()

# Fault-injection runs use the 8-context condition (one change each, same seeds).
FAULT_N_TRAIN = 8
FAULTS = {
    "correct": BASE,
    "gae_ignores_done": BASE._replace(faults=Faults(gae_ignores_done=True)),
    "truncation_as_termination": BASE._replace(faults=Faults(truncation_as_termination=True)),
    "no_adv_norm": BASE._replace(faults=Faults(no_adv_norm=True)),
    "lr_1e-2": BASE._replace(lr=1e-2, anneal_lr=False),
}

LENGTH_SWEEP = (0.08, 0.15, 0.25, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
FORCE_SWEEP = (2.0, 3.0, 5.0, 7.0, 10.0, 13.0, 18.0, 25.0, 35.0)
