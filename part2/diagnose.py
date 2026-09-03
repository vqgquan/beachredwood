"""
diagnose.py
Checks WHAT a trained agent actually does, instead of just its score.
Useful when an agent scores badly and you need to know why.

    python diagnose.py --scheme direct
"""

import argparse
import os
import numpy as np

from stable_baselines3 import PPO, DQN
from arena_env import ArenaEnv, WIDTH, HEIGHT

ACTION_NAMES = {
    "rotation": ["noop", "thrust", "rot_left", "rot_right", "shoot"],
    "direct": ["noop", "up", "down", "left", "right", "shoot"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", choices=["rotation", "direct"], default="rotation")
    p.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args()

    path = os.path.join("models", f"{args.algo}_{args.scheme}")
    model = (PPO if args.algo == "ppo" else DQN).load(path, device="cpu")
    env = ArenaEnv(scheme=args.scheme)
    names = ACTION_NAMES[args.scheme]

    counts = np.zeros(len(names))
    rewards, kills, spawners, phases, lengths, edge = [], [], [], [], [], []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=2000 + ep)
        total, n, near_wall = 0.0, 0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            counts[int(action)] += 1
            obs, r, term, trunc, info = env.step(action)
            total += r
            n += 1
            # is the ship hugging the border? (the classic "hide in a corner" failure)
            if min(env.px, WIDTH - env.px, env.py, HEIGHT - env.py) < 80:
                near_wall += 1
            if term or trunc:
                break
        rewards.append(total); kills.append(info["kills"])
        spawners.append(info["spawners_destroyed"]); phases.append(info["phase"])
        lengths.append(n); edge.append(near_wall / n)

    print(f"\n--- {args.algo}_{args.scheme} over {args.episodes} episodes ---")
    print(f"reward        {np.mean(rewards):8.2f} +/- {np.std(rewards):.2f}")
    print(f"episode len   {np.mean(lengths):8.1f} / {env.max_steps}")
    print(f"enemy kills   {np.mean(kills):8.2f}")
    print(f"spawner kills {np.mean(spawners):8.2f}")
    print(f"phase reached {np.mean(phases):8.2f}")
    print(f"time near wall{np.mean(edge) * 100:7.1f}%   (>60% suggests corner-hiding)")

    print("\naction usage:")
    for name, c in zip(names, counts / counts.sum() * 100):
        print(f"  {name:<10} {c:5.1f}%")

    # simple automatic warnings
    print()
    if counts[names.index("shoot")] / counts.sum() < 0.05:
        print("WARNING: the agent barely shoots -> it never found the combat reward.")
    if np.mean(edge) > 0.6 and np.mean(spawners) < 1:
        print("WARNING: corner-hiding policy -> avoidance beats fighting. Train longer.")
    if counts.max() / counts.sum() > 0.8:
        print("WARNING: one action dominates -> policy collapse. Raise ent_coef.")


if __name__ == "__main__":
    main()
