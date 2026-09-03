"""
part3/seed_check.py

Robustness check for the Part III ablation's network-capacity conclusion.

Why this exists
---------------
The ablation in part3/results/ was run with a single seed (42). Re-running the
`bigger_network` configuration later, byte-identical in configuration, produced
the opposite result:

    part3/results/ablation_results.csv (27/08)   bigger_network   -4.65 +/- 0.39
    part2/results/tuning_results.csv  (03/09)    net_big         -22.20 +/- 0.33

The baseline reproduced exactly across those two runs (-14.05 +/- 9.36 both
times), so this is not a code change. It points at PPO on this task being
knife-edge around its ~700k-step breakthrough: whether a run breaks through at
all can hinge on the random initialisation, which differs across library
versions even at a fixed seed.

A single-seed ablation cannot distinguish "this design choice is better" from
"this seed got lucky". This script re-runs the contested configurations across
several seeds so the report can state which conclusions actually hold.

    python part3/seed_check.py --seeds 43 44 --timesteps 800000

Results merge into results/seed_check.csv, so seeds can be added incrementally.
Combine with the seed-42 rows already in tuning_results.csv for the full picture.
"""

import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "part2"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from training_common import build_ppo, build_vec_envs

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# The three configurations whose conclusions depend on not being a fluke.
CONFIGS = [
    dict(name="baseline",   net_arch=(64, 64),   lr=3e-4, ent_coef=0.005),
    dict(name="net_big",    net_arch=(128, 128), lr=3e-4, ent_coef=0.005),
    dict(name="lr_high",    net_arch=(64, 64),   lr=1e-3, ent_coef=0.005),
]


def run_one(cfg, seed, timesteps, n_envs, eval_episodes):
    tag = f"{cfg['name']}_s{seed}"
    print(f"\n=== [{tag}] net={cfg['net_arch']} lr={cfg['lr']} seed={seed} ===",
          flush=True)

    train_env, eval_env = build_vec_envs("direct", n_envs=n_envs, seed=seed)
    model = build_ppo(train_env, net_arch=cfg["net_arch"],
                      learning_rate=cfg["lr"], ent_coef=cfg["ent_coef"],
                      seed=seed)

    best_dir = os.path.join(RESULTS_DIR, "seed_models", tag)
    os.makedirs(best_dir, exist_ok=True)
    cb = EvalCallback(eval_env, best_model_save_path=best_dir,
                      log_path=os.path.join(RESULTS_DIR, "seed_logs", tag),
                      eval_freq=max(20_000 // n_envs, 1),
                      n_eval_episodes=10, deterministic=True)

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)
    train_time = time.time() - t0

    best = os.path.join(best_dir, "best_model.zip")
    scored = PPO.load(best) if os.path.exists(best) else model

    fair_env, _ = build_vec_envs("direct", n_envs=1, shaping_enabled=False,
                                 seed=seed + 2000)
    mean_r, std_r = evaluate_policy(scored, fair_env,
                                    n_eval_episodes=eval_episodes,
                                    deterministic=True)
    train_env.close(); eval_env.close(); fair_env.close()

    print(f"    -> {tag}: {mean_r:+.2f} +/- {std_r:.2f} "
          f"({train_time/60:.1f} min)", flush=True)
    return {
        "config": cfg["name"],
        "seed": seed,
        "net_arch": "x".join(str(x) for x in cfg["net_arch"]),
        "learning_rate": cfg["lr"],
        "timesteps": timesteps,
        "mean_reward": round(mean_r, 2),
        "std_reward": round(std_r, 2),
        "train_time_sec": round(train_time, 1),
    }


def summarise(rows):
    print(f"\n{'config':12} {'n':>2}  {'seeds':16} {'mean':>8} {'spread':>16}")
    print("-" * 62)
    by = {}
    for r in rows:
        by.setdefault(r["config"], []).append(r)
    for name, rs in by.items():
        vals = [float(r["mean_reward"]) for r in rs]
        seeds = ",".join(str(r["seed"]) for r in rs)
        spread = f"{min(vals):+.2f} .. {max(vals):+.2f}"
        avg = statistics.mean(vals)
        print(f"{name:12} {len(vals):>2}  {seeds:16} {avg:+8.2f} {spread:>16}")
    print("\nIf a config's spread straddles both the ~-22 plateau and a "
          "broken-through score,\nthat config's single-seed result is not "
          "evidence of anything.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[43, 44])
    ap.add_argument("--timesteps", type=int, default=800_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "seed_check.csv")

    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    todo = [c for c in CONFIGS if args.only is None or c["name"] == args.only]
    done = {(r["config"], int(r["seed"])) for r in rows}

    for seed in args.seeds:
        for cfg in todo:
            if (cfg["name"], seed) in done:
                print(f"[seed_check] skip {cfg['name']} seed={seed} (already in CSV)")
                continue
            rows.append(run_one(cfg, seed, args.timesteps,
                                args.n_envs, args.eval_episodes))
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

    print(f"\n[seed_check] wrote {csv_path}")
    summarise(rows)


if __name__ == "__main__":
    main()
