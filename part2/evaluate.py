"""
evaluate.py
Loads a trained model and plays it in the visible arena.

Examples:
    python evaluate.py --scheme rotation --episodes 3
    python evaluate.py --scheme direct --model models/ppo_direct.zip
    python evaluate.py --scheme rotation --no-render     (fast, stats only)
"""

import argparse
import os
import numpy as np

from stable_baselines3 import PPO, DQN
from arena_env import ArenaEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", choices=["rotation", "direct"], default="rotation")
    p.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    p.add_argument("--model", default=None, help="path to a .zip model")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--no-render", action="store_true", help="run without a window")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    args = p.parse_args()

    path = args.model or os.path.join("models", f"{args.algo}_{args.scheme}")
    model = (PPO if args.algo == "ppo" else DQN).load(path)
    print(f"Loaded {path}")

    env = ArenaEnv(scheme=args.scheme, render_mode=None if args.no_render else "human")

    returns, phases = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
        phases.append(info["phase"])
        print(f"Episode {ep + 1}: reward {total:7.2f} | phase {info['phase']} | "
              f"kills {info['kills']} | spawners {info['spawners_destroyed']} | "
              f"{'died' if terminated else 'survived'}")

    env.close()
    print(f"\nMean reward {np.mean(returns):.2f} +/- {np.std(returns):.2f} | "
          f"mean phase {np.mean(phases):.2f}")


if __name__ == "__main__":
    main()
