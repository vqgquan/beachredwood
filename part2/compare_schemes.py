"""
compare_schemes.py
Reproducible side-by-side evaluation of the rotation and direct models,
for the report's "Comparison of Control Sets" section.

Runs BOTH models under IDENTICAL conditions (same number of episodes,
same fixed seed range, deterministic actions) so the resulting numbers
are a fair comparison rather than two separately-run, differently-lucky
evaluations. Reports mean +/- std, not just a point estimate, since this
environment has high episode-to-episode variance (std ~40-50 observed).

Usage:
    python compare_schemes.py --episodes 30
    python compare_schemes.py --episodes 30 --rotation-model models/ppo_rotation.zip --direct-model models/ppo_direct.zip

Writes control_set_comparison.csv, and prints a table in the same shape
as the report draft.
"""

import argparse
import csv
import numpy as np

from stable_baselines3 import PPO
from arena_env import ArenaEnv, WIDTH, HEIGHT

ACTION_NAMES = {
    "rotation": ["noop", "thrust", "rot_left", "rot_right", "shoot"],
    "direct": ["noop", "up", "down", "left", "right", "shoot"],
}


def evaluate(model_path, scheme, n_episodes, seed_start):
    """Runs n_episodes deterministic episodes and returns a dict of
    mean/std for every metric the report table needs, plus action usage
    (borrowed from diagnose.py) so you can also state whether either
    agent is exhibiting a degenerate policy."""
    model = PPO.load(model_path, device="cpu")
    env = ArenaEnv(scheme=scheme)
    names = ACTION_NAMES[scheme]
    counts = np.zeros(len(names))

    rewards, kills, spawners, phases, lengths, near_wall_frac = [], [], [], [], [], []

    for ep in range(n_episodes):
        # IMPORTANT: seeds are fixed and identical in role (not value) across
        # both schemes -- same seed_start, same n_episodes, same increment --
        # so both models face the same distribution of arena layouts.
        obs, info = env.reset(seed=seed_start + ep)
        total, steps, near_wall = 0.0, 0, 0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            counts[int(action)] += 1
            obs, r, terminated, truncated, info = env.step(int(action))
            total += r
            steps += 1
            if min(env.px, WIDTH - env.px, env.py, HEIGHT - env.py) < 80:
                near_wall += 1
        rewards.append(total)
        kills.append(info["kills"])
        spawners.append(info["spawners_destroyed"])
        phases.append(info["phase"])
        lengths.append(steps)
        near_wall_frac.append(near_wall / steps)

    action_usage = {name: pct for name, pct in
                     zip(names, counts / counts.sum() * 100)}

    return {
        "scheme": scheme,
        "n_episodes": n_episodes,
        "reward_mean": np.mean(rewards), "reward_std": np.std(rewards),
        "kills_mean": np.mean(kills), "kills_std": np.std(kills),
        "spawners_mean": np.mean(spawners), "spawners_std": np.std(spawners),
        "phase_mean": np.mean(phases), "phase_std": np.std(phases),
        "length_mean": np.mean(lengths),
        "near_wall_pct": np.mean(near_wall_frac) * 100,
        "action_usage": action_usage,
    }


def print_report_table(rot, direct):
    def fmt(m, s):
        return f"{m:+.1f} +/- {s:.1f}"

    print(f"\n{'Metric':<28}{'Rotation':<22}{'Direct':<22}")
    print("-" * 72)
    print(f"{'Episodes evaluated':<28}{rot['n_episodes']:<22}{direct['n_episodes']:<22}")
    print(f"{'Mean episode reward':<28}{fmt(rot['reward_mean'], rot['reward_std']):<22}{fmt(direct['reward_mean'], direct['reward_std']):<22}")
    print(f"{'Enemy kills':<28}{fmt(rot['kills_mean'], rot['kills_std']):<22}{fmt(direct['kills_mean'], direct['kills_std']):<22}")
    print(f"{'Spawners destroyed':<28}{fmt(rot['spawners_mean'], rot['spawners_std']):<22}{fmt(direct['spawners_mean'], direct['spawners_std']):<22}")
    print(f"{'Phase reached':<28}{fmt(rot['phase_mean'], rot['phase_std']):<22}{fmt(direct['phase_mean'], direct['phase_std']):<22}")
    print(f"{'Mean episode length':<28}{rot['length_mean']:<22.1f}{direct['length_mean']:<22.1f}")
    print(f"{'Time near wall':<28}{rot['near_wall_pct']:<22.1f}{direct['near_wall_pct']:<22.1f}")
    print()
    print("Action usage (rotation):", {k: round(v, 1) for k, v in rot["action_usage"].items()})
    print("Action usage (direct):  ", {k: round(v, 1) for k, v in direct["action_usage"].items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30,
                    help="More episodes = tighter std. 30 is a reasonable minimum given the observed variance.")
    p.add_argument("--seed-start", type=int, default=9000,
                    help="Use a seed range that was NEVER used during training or tune.py, for a clean held-out evaluation.")
    p.add_argument("--rotation-model", default="models/ppo_rotation.zip")
    p.add_argument("--direct-model", default="models/ppo_direct.zip")
    args = p.parse_args()

    print(f"Evaluating both schemes over {args.episodes} deterministic episodes "
          f"(seeds {args.seed_start}-{args.seed_start + args.episodes - 1})...")

    rot = evaluate(args.rotation_model, "rotation", args.episodes, args.seed_start)
    direct = evaluate(args.direct_model, "direct", args.episodes, args.seed_start)

    print_report_table(rot, direct)

    with open("control_set_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "rotation_mean", "rotation_std", "direct_mean", "direct_std"])
        for key in ["reward", "kills", "spawners", "phase"]:
            w.writerow([key, rot[f"{key}_mean"], rot[f"{key}_std"],
                        direct[f"{key}_mean"], direct[f"{key}_std"]])
    print("\nSaved control_set_comparison.csv")


if __name__ == "__main__":
    main()
