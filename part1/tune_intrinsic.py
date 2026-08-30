#!/usr/bin/env python3
"""Hyperparameter sweep over intrinsicRewardStrength for Level 6 (Task 5).

config.json ships with intrinsicRewardStrength = 0.2. generate_evidence.py's
main run shows this is too large relative to Level 6's own +1/+2 rewards: it
biases Q-learning toward under-visited states instead of actually finishing
the level (lower success rate than with no intrinsic reward at all). This
script sweeps a small set of strengths and reports the success rate / final
return for each, to justify a better-tuned value for the report.

Run with:
    python tune_intrinsic.py
"""
import csv
import os

from generate_evidence import TRAINING_CURVES_DIR, train

STRENGTHS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
ALGOS = ["q", "sarsa"]


def main():
    rows = []
    for algo in ALGOS:
        for strength in STRENGTHS:
            label = f"level6_{algo}_strength{strength}"
            print(f"Training {label} ...", end=" ", flush=True)
            _, metrics = train(6, algo, intrinsic_enabled=True, intrinsic_strength=strength)
            episodes = metrics["episodes"]
            last100 = metrics["env_return_history"][-100:]
            row = {
                "algorithm": algo,
                "intrinsic_strength": strength,
                "success_rate": round(metrics["successes"] / max(1, episodes), 4),
                "avg_env_return_last100": round(sum(last100) / max(1, len(last100)), 3),
                "deaths": metrics["deaths"],
                "timeouts": metrics["timeouts"],
            }
            rows.append(row)
            print(f"success_rate={row['success_rate']:.2f} "
                  f"last100={row['avg_env_return_last100']:.2f}")

    out = os.path.join(TRAINING_CURVES_DIR, "intrinsic_strength_sweep.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
