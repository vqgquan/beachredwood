#!/usr/bin/env python3
"""Hyperparameter sweep over alpha, gamma and epsilonDecayEpisodes (report
section 4, Part I table).

tune_intrinsic.py already covers intrinsicRewardStrength. This script covers
the three learning hyperparameters config.json ships, so the report can say
why each shipped value was chosen rather than presenting them unexplained.

Method: one axis at a time away from the shipped configuration, everything
else held fixed, same seed. Run on Level 2 -- it has multiple apples plus a
key/chest sequence (a real planning problem) and no monsters, so results are
not blurred by the stochastic monster movement.

Run with:
    python tune_hyperparams.py
"""
import csv
import os

import generate_evidence as ge
from generate_evidence import TRAINING_CURVES_DIR

LEVEL = 2
ALGOS = ["q", "sarsa"]

BASE = dict(alpha=ge.ALPHA, gamma=ge.GAMMA, eps_decay=ge.EPS_DECAY_EP)

SWEEPS = [
    ("alpha",     [0.1, 0.2, 0.4]),
    ("gamma",     [0.90, 0.95, 0.99]),
    ("eps_decay", [400, 700, 1000]),
]


def run(algo, alpha, gamma, eps_decay):
    """Patch the module globals train() reads, then train one config."""
    ge.ALPHA, ge.GAMMA, ge.EPS_DECAY_EP = alpha, gamma, eps_decay
    _, m = ge.train(LEVEL, algo, intrinsic_enabled=False)
    eps_n = m["episodes"]
    last100 = m["env_return_history"][-100:]
    succ = m["successes"]
    return {
        "success_rate": round(succ / max(1, eps_n), 4),
        "avg_env_return_last100": round(sum(last100) / max(1, len(last100)), 3),
        "avg_steps_on_success": round(m["successful_steps"] / max(1, succ), 1),
        "deaths": m["deaths"],
        "timeouts": m["timeouts"],
        "q_table_states": len(m.get("q_states", [])) or "",
    }


def main():
    rows = []
    for axis, values in SWEEPS:
        for algo in ALGOS:
            for v in values:
                cfg = dict(BASE)
                cfg[axis] = v
                label = f"{axis}={v} {algo}"
                print(f"  {label:26}", end=" ", flush=True)
                r = run(algo, cfg["alpha"], cfg["gamma"], cfg["eps_decay"])
                rows.append(dict(
                    axis=axis, algorithm=algo, value=v,
                    is_shipped=(v == BASE[axis]), **r))
                print(f"success={r['success_rate']:.3f} "
                      f"return={r['avg_env_return_last100']:.2f} "
                      f"steps={r['avg_steps_on_success']}")
    ge.ALPHA, ge.GAMMA, ge.EPS_DECAY_EP = BASE["alpha"], BASE["gamma"], BASE["eps_decay"]

    out = os.path.join(TRAINING_CURVES_DIR, "hyperparam_sweep.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
