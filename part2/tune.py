"""
tune.py
Small hyperparameter sweep: trains a few short PPO runs and compares them.
Results are printed and written to tuning_results.csv.

Example:
    python tune.py --scheme rotation --timesteps 60000
"""

import argparse
import csv
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from arena_env import ArenaEnv

# each dict is one configuration to test
CONFIGS = [
    {"name": "baseline_defaults", "learning_rate": 3e-4, "n_steps": 2048,
     "ent_coef": 0.0, "gamma": 0.99, "net": [64, 64]},
    {"name": "high_entropy", "learning_rate": 3e-4, "n_steps": 1024,
     "ent_coef": 0.01, "gamma": 0.99, "net": [64, 64]},
    {"name": "high_gamma_bignet", "learning_rate": 3e-4, "n_steps": 1024,
     "ent_coef": 0.01, "gamma": 0.995, "net": [128, 128]},
    {"name": "fast_lr", "learning_rate": 1e-3, "n_steps": 1024,
     "ent_coef": 0.01, "gamma": 0.995, "net": [128, 128]},
]


def run(cfg, scheme, timesteps, n_envs, seed):
    """Train one configuration briefly and return its mean evaluation reward."""
    env = make_vec_env(ArenaEnv, n_envs=n_envs, seed=seed, env_kwargs={"scheme": scheme})
    model = PPO("MlpPolicy", env,
                learning_rate=cfg["learning_rate"],
                n_steps=cfg["n_steps"],
                batch_size=256,
                ent_coef=cfg["ent_coef"],
                gamma=cfg["gamma"],
                policy_kwargs=dict(net_arch=dict(pi=cfg["net"], vf=cfg["net"])),
                tensorboard_log="tensorboard",
                seed=seed, verbose=0)
    model.learn(total_timesteps=timesteps, tb_log_name=f"tune_{scheme}_{cfg['name']}")

    eval_env = make_vec_env(ArenaEnv, n_envs=1, seed=seed + 100,
                            env_kwargs={"scheme": scheme})
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)
    return mean_r, std_r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", choices=["rotation", "direct"], default="rotation")
    p.add_argument("--timesteps", type=int, default=60_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = []
    for cfg in CONFIGS:
        print(f"--- training {cfg['name']} ---")
        mean_r, std_r = run(cfg, args.scheme, args.timesteps, args.n_envs, args.seed)
        print(f"{cfg['name']}: {mean_r:.2f} +/- {std_r:.2f}")
        rows.append({**cfg, "scheme": args.scheme,
                     "mean_reward": round(mean_r, 2), "std_reward": round(std_r, 2)})

    with open("tuning_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    best = max(rows, key=lambda r: r["mean_reward"])
    print(f"\nBest config: {best['name']} ({best['mean_reward']}) -> saved tuning_results.csv")


if __name__ == "__main__":
    main()
