#!/usr/bin/env python3
"""Headless evidence generator for the Part I report.

Runs the exact same Q-learning / SARSA training that app.py performs
interactively, but without opening a Pygame window, so the report evidence
(training curves + comparisons) can be regenerated in one deterministic pass
instead of manually driving the interactive app through every level.

For every level (0-6) and both algorithms, and both intrinsic settings on
Level 6, this writes:

  training_curves/level{N}_{algo}_{intrinsic|extrinsic}.csv
      per-episode (env_return, total_return, status) -- same format app.py
      writes, used directly by plot_curves.py.

  training_curves/paths/level{N}_{algo}_{intrinsic|extrinsic}.json
      a single greedy (epsilon=0) rollout after training: the policy's
      shortest/learned path, used as B5/C3 evidence.

  training_curves/summary.csv
      one row per run with success rate, deaths by cause, and average return
      -- the numbers used in the report tables.

Run with:
    python generate_evidence.py
"""
import csv
import json
import math
import os

from agents import (QTable, epsilon_greedy, greedy_action, linear_epsilon,
                     q_learning_update, sarsa_update)
from config import (ALPHA, EPISODES, EPS_DECAY_EP, EPS_END, EPS_START, GAMMA,
                     INTRINSIC_REWARD_STRENGTH, MAX_STEPS, STEP_PENALTY)
from env import GridWorld
from levels import LEVELS
from metrics import make_metrics, record_episode, save_training_curve

TRAINING_CURVES_DIR = os.path.join(os.path.dirname(__file__), "training_curves")
PATHS_DIR = os.path.join(TRAINING_CURVES_DIR, "paths")


def train(level_idx, algo_name, intrinsic_enabled, intrinsic_strength=None):
    """Train one Q-table on one level with one algorithm, EPISODES episodes.

    Mirrors app.py's per-step training loop exactly (same functions, same
    reward composition) so headless and interactive runs are comparable.
    `intrinsic_strength` overrides config.json's value (used by
    tune_intrinsic.py); leave None to use the configured strength.
    """
    strength = INTRINSIC_REWARD_STRENGTH if intrinsic_strength is None else intrinsic_strength
    env = GridWorld(LEVELS[level_idx])
    qtab = QTable()
    metrics = make_metrics()

    for ep in range(EPISODES):
        eps = linear_epsilon(ep, EPS_START, EPS_END, EPS_DECAY_EP)
        s = env.reset()
        a = epsilon_greedy(qtab, s, eps) if algo_name == "sarsa" else None
        visit_counts = {}
        total_reward = 0.0
        intrinsic_episode = 0.0
        env_reward_episode = 0.0
        steps = 0
        res = None

        while True:
            visits = visit_counts.get(s, 0) + 1
            visit_counts[s] = visits
            intrinsic = 0.0
            if level_idx == 6 and intrinsic_enabled:
                intrinsic = strength / math.sqrt(visits + 1)

            if algo_name == "q":
                a = epsilon_greedy(qtab, s, eps)
                res = env.step(a)
                learning_reward = res.reward + intrinsic + STEP_PENALTY
                q_learning_update(qtab, s, a, learning_reward, res.next_state,
                                  ALPHA, GAMMA, res.done)
                s = res.next_state
            else:
                res = env.step(a)
                ap = epsilon_greedy(qtab, res.next_state, eps)
                learning_reward = res.reward + intrinsic + STEP_PENALTY
                sarsa_update(qtab, s, a, learning_reward, res.next_state, ap,
                             ALPHA, GAMMA, res.done)
                s = res.next_state
                a = ap

            total_reward += learning_reward
            intrinsic_episode += intrinsic
            env_reward_episode += res.reward
            steps += 1
            if res.done or steps >= MAX_STEPS:
                break

        record_episode(metrics, steps, total_reward, intrinsic_episode,
                        env_reward_episode, res.done, env.alive, res.info)

    return qtab, metrics


def rollout_path(level_idx, qtab):
    """One purely-greedy (no exploration) rollout, for path-evidence plots."""
    env = GridWorld(LEVELS[level_idx])
    s = env.reset()
    path = [env.agent]
    done = False
    for _ in range(MAX_STEPS):
        a = greedy_action(qtab, s)
        res = env.step(a)
        path.append(env.agent)
        s = res.next_state
        done = res.done
        if done:
            break
    return {"path": [list(p) for p in path], "alive": env.alive, "done": done}


def build_run_list():
    runs = []
    for level in sorted(LEVELS):
        for algo in ("q", "sarsa"):
            if level == 6:
                runs.append((level, algo, True))
                runs.append((level, algo, False))
            else:
                runs.append((level, algo, False))
    return runs


def main():
    os.makedirs(PATHS_DIR, exist_ok=True)
    summary_rows = []

    for level, algo, intrinsic in build_run_list():
        label = f"level{level}_{algo}_{'intrinsic' if intrinsic else 'extrinsic'}"
        print(f"Training {label} ...", end=" ", flush=True)

        qtab, metrics = train(level, algo, intrinsic)
        save_training_curve(level, algo, metrics, intrinsic, TRAINING_CURVES_DIR)

        rollout = rollout_path(level, qtab)
        with open(os.path.join(PATHS_DIR, f"{label}.json"), "w", encoding="utf-8") as f:
            json.dump(rollout, f)

        episodes = metrics["episodes"]
        avg_return = metrics["return_total"] / max(1, episodes)
        avg_env_return = sum(metrics["env_return_history"]) / max(1, episodes)
        last100 = metrics["env_return_history"][-100:]
        avg_env_return_last100 = sum(last100) / max(1, len(last100))
        avg_steps_success = metrics["successful_steps"] / max(1, metrics["successes"])

        row = {
            "level": level,
            "algorithm": algo,
            "intrinsic": intrinsic,
            "episodes": episodes,
            "success_rate": round(metrics["successes"] / max(1, episodes), 4),
            "deaths": metrics["deaths"],
            "fire_deaths": metrics["fire_deaths"],
            "monster_deaths": metrics["monster_deaths"],
            "timeouts": metrics["timeouts"],
            "avg_return": round(avg_return, 3),
            "avg_env_return": round(avg_env_return, 3),
            "avg_env_return_last100": round(avg_env_return_last100, 3),
            "avg_steps_on_success": round(avg_steps_success, 2),
            "q_table_states": len(qtab.q),
            "rollout_alive": rollout["alive"],
            "rollout_steps": len(rollout["path"]) - 1,
        }
        summary_rows.append(row)
        print(f"success_rate={row['success_rate']:.2f} "
              f"avg_env_return_last100={row['avg_env_return_last100']:.2f}")

    summary_path = os.path.join(TRAINING_CURVES_DIR, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote {summary_path}")
    print("Run plot_curves.py to turn these into report-ready PNG charts.")


if __name__ == "__main__":
    main()
