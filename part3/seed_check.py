"""
part3/seed_check.py

Robustness check: re-runs the ablation arms across several seeds.

run_ablation.py runs one seed. That is enough to produce a table and not
enough to conclude from it. PPO can break through or fail to within the same
budget depending only on initialisation, so a single-seed gap between two arms
may be nothing but noise. This script re-runs the arms and reports the spread.

    python part3/seed_check.py --seeds 1 2

Read this before writing any sentence of the form "X is better than Y". If an
arm's spread overlaps the baseline's, the single-seed gap is not evidence.

Results merge into results/seed_check.csv so seeds can be added incrementally.
"""

import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from ablation_configs import ABLATION_CONFIGS          # noqa: E402
from run_ablation import RESULTS_DIR, run_one          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--timesteps", type=int, default=700_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "seed_check.csv")

    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    done = {(r["name"], int(r["seed"])) for r in rows}

    # Seed 0 already exists in ablation_results.csv -- fold it in so the
    # summary counts every run, not only the ones this script produced.
    base_csv = os.path.join(RESULTS_DIR, "ablation_results.csv")
    if os.path.exists(base_csv):
        with open(base_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r["name"], int(r["seed"])) not in done:
                    rows.append(r)
                    done.add((r["name"], int(r["seed"])))

    for seed in args.seeds:
        for cfg in ABLATION_CONFIGS:
            if (cfg["name"], seed) in done:
                print(f"[seed_check] skip {cfg['name']} seed={seed}")
                continue
            rows.append(run_one(cfg, args.timesteps, args.n_envs,
                                args.episodes, seed))
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

    print(f"\n[seed_check] wrote {csv_path}\n")
    by = {}
    for r in rows:
        by.setdefault(r["name"], []).append(float(r["mean_reward"]))
    order = [c["name"] for c in ABLATION_CONFIGS]
    base = statistics.mean(by["baseline"]) if "baseline" in by else None
    print(f"{'config':15} {'n':>2} {'mean':>9} {'spread':>20}  vs baseline")
    print("-" * 66)
    for name in order:
        v = by.get(name, [])
        if not v:
            continue
        m = statistics.mean(v)
        d = "" if name == "baseline" or base is None else f"{m - base:+.1f}"
        print(f"{name:15} {len(v):>2} {m:+9.1f} "
              f"{min(v):+9.1f} .. {max(v):+7.1f}  {d}")
    print("\nAn arm whose spread overlaps the baseline's is not distinguishable "
          "from it at this number of seeds.")


if __name__ == "__main__":
    main()
