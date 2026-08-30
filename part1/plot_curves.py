#!/usr/bin/env python3
"""Turn training_curves/*.csv and paths/*.json into report-ready PNG charts.

Produces, under training_curves/plots/:
  level{N}_q_vs_sarsa.png            -- Task 1-4 evidence (learning curves)
  level6_{algo}_intrinsic_vs_extrinsic.png -- Task 5 evidence
  paths/level{N}_{algo}_{variant}.png       -- learned greedy path on the grid
  level1_q_vs_sarsa_path.png                -- Task 2 "more conservative" evidence

Run after generate_evidence.py:
    python plot_curves.py
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from levels import LEVELS, LEVEL_LABELS

TRAINING_CURVES_DIR = os.path.join(os.path.dirname(__file__), "training_curves")
PATHS_DIR = os.path.join(TRAINING_CURVES_DIR, "paths")
PLOTS_DIR = os.path.join(TRAINING_CURVES_DIR, "plots")

CELL_COLORS = {
    "R": "#5a5f6e",
    "F": "#f0691f",
    "A": "#fc5c65",
    "K": "#f7d23f",
    "C": "#b47850",
    "M": "#aa46d2",
}


def moving_average(values, window=25):
    if not values:
        return []
    out = []
    running = 0.0
    q = []
    for v in values:
        q.append(v)
        running += v
        if len(q) > window:
            running -= q.pop(0)
        out.append(running / len(q))
    return out


def read_curve(level, algo, intrinsic_label):
    path = os.path.join(TRAINING_CURVES_DIR, f"level{level}_{algo}_{intrinsic_label}.csv")
    if not os.path.exists(path):
        return None
    episodes, env_returns = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            env_returns.append(float(row["env_return"]))
    return episodes, env_returns


def plot_algorithm_comparison():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for level in sorted(LEVELS):
        q_curve = read_curve(level, "q", "intrinsic" if level == 6 else "extrinsic")
        s_curve = read_curve(level, "sarsa", "intrinsic" if level == 6 else "extrinsic")
        if q_curve is None and s_curve is None:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))
        if q_curve is not None:
            ep, ret = q_curve
            ax.plot(ep, moving_average(ret), label="Q-learning", color="#4a90d9")
        if s_curve is not None:
            ep, ret = s_curve
            ax.plot(ep, moving_average(ret), label="SARSA", color="#e0724a")
        ax.set_title(f"{LEVEL_LABELS.get(level, f'Level {level}')} -- environment return "
                     f"(25-episode moving average)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Environment return")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"level{level}_q_vs_sarsa.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


def plot_intrinsic_comparison():
    for algo, color_extrinsic, color_intrinsic in (
        ("q", "#4a90d9", "#2ecc71"),
        ("sarsa", "#e0724a", "#2ecc71"),
    ):
        extrinsic = read_curve(6, algo, "extrinsic")
        intrinsic = read_curve(6, algo, "intrinsic")
        if extrinsic is None or intrinsic is None:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))
        ep, ret = extrinsic
        ax.plot(ep, moving_average(ret), label="Without intrinsic reward", color=color_extrinsic)
        ep, ret = intrinsic
        ax.plot(ep, moving_average(ret), label="With intrinsic reward", color=color_intrinsic)
        ax.set_title(f"Level 6 ({algo.upper()}) -- environment return, "
                     f"with vs without intrinsic reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Environment return (intrinsic reward excluded)")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"level6_{algo}_intrinsic_vs_extrinsic.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


def draw_grid_background(ax, level_idx):
    layout = LEVELS[level_idx]
    h, w = len(layout), len(layout[0])
    for y, row in enumerate(layout):
        for x, ch in enumerate(row):
            if ch in CELL_COLORS:
                ax.add_patch(plt.Rectangle((x, h - 1 - y), 1, 1,
                                            color=CELL_COLORS[ch], alpha=0.9, zorder=1))
            elif ch == "S":
                ax.add_patch(plt.Rectangle((x, h - 1 - y), 1, 1,
                                            facecolor="none", edgecolor="#4ade80",
                                            linewidth=2, zorder=1))
    for x in range(w + 1):
        ax.axvline(x, color="#333333", linewidth=0.5, zorder=0)
    for y in range(h + 1):
        ax.axhline(y, color="#333333", linewidth=0.5, zorder=0)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return h


def path_to_xy(path, h):
    xs = [p[0] + 0.5 for p in path]
    ys = [h - 1 - p[1] + 0.5 for p in path]
    return xs, ys


def load_path(label):
    path_file = os.path.join(PATHS_DIR, f"{label}.json")
    if not os.path.exists(path_file):
        return None
    with open(path_file, encoding="utf-8") as f:
        return json.load(f)


def plot_individual_paths():
    out_dir = os.path.join(PLOTS_DIR, "paths")
    os.makedirs(out_dir, exist_ok=True)
    for level in sorted(LEVELS):
        for algo in ("q", "sarsa"):
            variants = ["intrinsic", "extrinsic"] if level == 6 else ["extrinsic"]
            for variant in variants:
                label = f"level{level}_{algo}_{variant}"
                data = load_path(label)
                if data is None:
                    continue
                fig, ax = plt.subplots(figsize=(5, 3.5))
                h = draw_grid_background(ax, level)
                xs, ys = path_to_xy(data["path"], h)
                ax.plot(xs, ys, color="#4ade80", linewidth=2, marker="o", markersize=3,
                        zorder=2)
                ax.plot(xs[0], ys[0], marker="s", color="#4ade80", markersize=10, zorder=3)
                outcome = "reached goal" if data["done"] and data["alive"] else (
                    "died" if not data["alive"] else "timed out")
                ax.set_title(f"Level {level} greedy rollout -- {algo.upper()} "
                             f"({variant}, {outcome}, {len(xs) - 1} steps)")
                fig.tight_layout()
                out = os.path.join(out_dir, f"{label}.png")
                fig.savefig(out, dpi=150)
                plt.close(fig)
                print(f"wrote {out}")


def plot_level1_comparison():
    """Task 2 (C3) evidence: overlay the Q-learning and SARSA greedy paths on
    Level 1, where the shortest route runs between two rows of fire."""
    q_data = load_path("level1_q_extrinsic")
    s_data = load_path("level1_sarsa_extrinsic")
    if q_data is None or s_data is None:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    h = draw_grid_background(ax, 1)
    qx, qy = path_to_xy(q_data["path"], h)
    sx, sy = path_to_xy(s_data["path"], h)
    ax.plot(qx, qy, color="#4a90d9", linewidth=3, marker="o", markersize=4,
            label="Q-learning (off-policy)", zorder=2)
    ax.plot(sx, sy, color="#e0724a", linewidth=3, linestyle="--", marker="o", markersize=4,
            label="SARSA (on-policy)", zorder=2, alpha=0.9)
    ax.set_title("Level 1 -- learned greedy path, Q-learning vs SARSA")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "level1_q_vs_sarsa_path.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_algorithm_comparison()
    plot_intrinsic_comparison()
    plot_individual_paths()
    plot_level1_comparison()


if __name__ == "__main__":
    main()
