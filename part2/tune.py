"""
part2/tune.py

Hyperparameter exploration for the Part II PPO agent (rubric J3: "tuned in a
meaningful way, not default-only, with some evidence of exploration").

Sweeps one hyperparameter at a time away from the shipped baseline and scores
each variant on TASK-ONLY reward, so the numbers in the report's hyperparameter
table are comparable to each other and to the Part III ablation table.

    python tune.py --timesteps 800000 --n-envs 8

Writes results/tuning_results.csv and prints a summary table.

Why 800k timesteps and not a quick 100k: PPO on this arena plateaus for roughly
700k steps before breaking through (see part2/README.md, "Real training
results"). A short sweep would show every configuration sitting flat on the same
plateau and would be evidence of nothing. The sweep has to outlast the plateau
to say anything.

Protocol, mirroring part3/run_ablation.py so the two tables agree:
  - one axis changed per run, everything else identical, same seed
  - EvalCallback tracks the BEST checkpoint during training, because the
    final-step policy is often worse than the best one (direct's final model
    scores -22.8 while its step-740k checkpoint scores -1.45)
  - final scoring uses a fresh env with shaping DISABLED, so a config is never
    credited for its own shaping bonus
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from training_common import SEED, build_ppo, build_vec_envs

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# The configuration currently shipped in training_common.build_ppo.
BASELINE = dict(
    name="baseline",
    axis="-",
    control_scheme="direct",
    lr=3e-4,
    net_arch=(64, 64),
    ent_coef=0.005,
)

# Each variant moves exactly one knob. `note` records what the run is meant to
# answer, so the report's "effect observed" column has something to build on.
CONFIGS = [
    BASELINE,
    dict(BASELINE, name="lr_low", axis="learning_rate", lr=1e-4,
         note="Is 3e-4 too fast? A smaller step should be steadier but slower."),
    dict(BASELINE, name="lr_high", axis="learning_rate", lr=1e-3,
         note="Is 3e-4 too slow? A larger step may break through sooner or diverge."),
    dict(BASELINE, name="ent_default", axis="ent_coef", ent_coef=0.0,
         note="ent_coef=0.005 is non-default. Does the extra exploration bonus "
              "actually earn its place, or would SB3's default 0.0 do as well?"),
    dict(BASELINE, name="net_big", axis="net_arch", net_arch=(128, 128),
         note="Capacity check. Part III studies this axis in depth; here it is "
              "only to complete the hyperparameter table."),
]


def run_one(cfg, timesteps, n_envs, eval_episodes, eval_freq=20_000):
    print(f"\n=== [{cfg['name']}] axis={cfg['axis']} lr={cfg['lr']} "
          f"net_arch={cfg['net_arch']} ent_coef={cfg['ent_coef']} ===", flush=True)

    train_env, eval_env = build_vec_envs(
        cfg["control_scheme"], n_envs=n_envs, seed=SEED,
    )
    model = build_ppo(
        train_env,
        net_arch=cfg["net_arch"],
        learning_rate=cfg["lr"],
        ent_coef=cfg["ent_coef"],
        seed=SEED,
    )

    best_dir = os.path.join(RESULTS_DIR, "models", cfg["name"])
    os.makedirs(best_dir, exist_ok=True)
    callback = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=os.path.join(RESULTS_DIR, "logs", cfg["name"]),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    train_time = time.time() - t0

    best_path = os.path.join(best_dir, "best_model.zip")
    scored = PPO.load(best_path) if os.path.exists(best_path) else model

    fair_env, _ = build_vec_envs(
        cfg["control_scheme"], n_envs=1, shaping_enabled=False, seed=SEED + 2000,
    )
    mean_r, std_r = evaluate_policy(
        scored, fair_env, n_eval_episodes=eval_episodes, deterministic=True
    )
    train_env.close(); eval_env.close(); fair_env.close()

    print(f"    -> {cfg['name']}: {mean_r:+.2f} +/- {std_r:.2f} "
          f"({train_time/60:.1f} min)", flush=True)

    return {
        "name": cfg["name"],
        "axis": cfg["axis"],
        "learning_rate": cfg["lr"],
        "net_arch": "x".join(str(x) for x in cfg["net_arch"]),
        "ent_coef": cfg["ent_coef"],
        "timesteps": timesteps,
        "mean_reward": round(mean_r, 2),
        "std_reward": round(std_r, 2),
        "train_time_sec": round(train_time, 1),
        "note": cfg.get("note", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=800_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--only", default=None,
                    help="run a single config by name, merging into the existing CSV")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "tuning_results.csv")

    rows = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = {r["name"]: r for r in csv.DictReader(f)}

    todo = [c for c in CONFIGS if args.only is None or c["name"] == args.only]
    if not todo:
        raise SystemExit(f"No config named {args.only!r}. "
                         f"Available: {[c['name'] for c in CONFIGS]}")

    for cfg in todo:
        rows[cfg["name"]] = run_one(
            cfg, args.timesteps, args.n_envs, args.eval_episodes
        )

    ordered = [rows[c["name"]] for c in CONFIGS if c["name"] in rows]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ordered[0].keys()))
        w.writeheader()
        w.writerows(ordered)

    print(f"\n[tune] wrote {csv_path}\n")
    base = next((r for r in ordered if r["name"] == "baseline"), None)
    print(f"{'config':14} {'axis':14} {'value':12} {'reward':>16}  vs baseline")
    print("-" * 74)
    for r in ordered:
        value = {"learning_rate": r["learning_rate"], "net_arch": r["net_arch"],
                 "ent_coef": r["ent_coef"]}.get(r["axis"], "-")
        delta = ""
        if base and r["name"] != "baseline":
            d = float(r["mean_reward"]) - float(base["mean_reward"])
            delta = f"{d:+.2f}"
        print(f"{r['name']:14} {r['axis']:14} {str(value):12} "
              f"{float(r['mean_reward']):+8.2f} +/- {float(r['std_reward']):5.2f}  {delta}")


if __name__ == "__main__":
    main()
