"""
part3/plot_ablation.py

Turns results/ablation_results.csv into the report figures.

    python part3/plot_ablation.py

Writes results/ablation_summary.png and, if results/seed_check.csv exists,
results/ablation_seeds.png.

Why three panels and not one: reward alone cannot distinguish an agent that
fights well from one that refuses to fight. Episode length and kills separate
those two, and on this study they are what the interesting arms turn on --
no_ship_frame survives longer than the baseline while destroying almost
nothing, which is a behaviour a reward bar would hide.
"""

import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

ACCENT, INK, MUTED = "#2A63C4", "#141922", "#6B7688"
GRID, SURF, BASECOL = "#E4E9EF", "#FCFCFB", "#8A94A6"

# Categorical hues for the three variants; baseline stays grey as the
# reference. Validated for CVD separation and contrast against the light
# chart surface. This figure targets a printed report, so it is light-only.
VARIANT = {
    "no_shaping": "#2A63C4",
    "no_step_cost": "#B45309",
    "no_ship_frame": "#A21CAF",
}

LABEL = {
    "baseline": "baseline",
    "no_shaping": "no_shaping\nR_SHAPE = 0",
    "no_step_cost": "no_step_cost\nR_STEP = 0",
    "no_ship_frame": "no_ship_frame\nobs[9,10,15,16] = 0",
}

PANELS = [
    ("mean_reward", "Task reward  (higher is better)"),
    ("mean_ep_len", "Episode length  (cap 1200 steps)"),
    ("mean_kills", "Enemies destroyed per episode"),
]


def _style(ax):
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def plot_summary(rows):
    names = [r["name"] for r in rows]
    y = range(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), sharey=True)

    for ax, (col, title) in zip(axes, PANELS):
        vals = [float(r[col]) for r in rows]
        base = next((float(r[col]) for r in rows if r["name"] == "baseline"), None)
        colors = [BASECOL if n == "baseline" else ACCENT for n in names]
        ax.barh(list(y), vals, height=.6, color=colors, zorder=3)
        if base is not None:
            ax.axvline(base, color=BASECOL, lw=1.2, ls=(0, (4, 3)), zorder=2)
        for i, v in enumerate(vals):
            off = 4 if v >= 0 else -4
            ax.annotate(f"{v:,.1f}", (v, i), xytext=(off, 0),
                        textcoords="offset points", va="center",
                        ha="left" if v >= 0 else "right",
                        fontsize=8.5, color=INK)
        span = max(vals) - min(min(vals), 0) or 1
        # leave room on the left for a label sitting outside a negative bar
        left = min(min(vals), 0) - (span * .22 if min(vals) < 0 else span * .04)
        ax.set_xlim(left, max(vals) + span * .22)
        ax.set_title(title, fontsize=9.5, fontweight="bold", color=INK,
                     loc="left", pad=10)
        _style(ax)

    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels([LABEL.get(n, n) for n in names],
                            fontsize=8.5, color=INK)
    axes[0].invert_yaxis()
    fig.tight_layout(pad=1.5)
    out = os.path.join(RESULTS_DIR, "ablation_summary.png")
    fig.savefig(out, dpi=200, facecolor=SURF)
    print(f"  wrote {out}")


def plot_curves(order):
    """Learning curves from each arm's EvalCallback log.

    The brief asks for training curves as well as final performance, and they
    answer a question the bar chart cannot: whether an arm learned slowly, hit
    a ceiling, or never left the ground. Baseline is grey to mark it as the
    reference, matching the summary figure.
    """
    import numpy as np
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    drawn = 0
    for name in order:
        path = os.path.join(RESULTS_DIR, "logs", name, "evaluations.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path)
        x, y = d["timesteps"], d["results"].mean(axis=1)
        col = BASECOL if name == "baseline" else VARIANT[name]
        ax.plot(x, y, color=col, lw=2, zorder=3,
                label=name.replace("_", " "))
        # direct end-label: the palette's green/orange pair sits in the CVD
        # floor band, which is only legal with a secondary encoding
        ax.annotate(name.replace("_", " "), (x[-1], y[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=col, fontweight="bold")
        drawn += 1
    if drawn == 0:
        print("  (no evaluations.npz -- run run_ablation.py to produce curves)")
        plt.close(fig)
        return

    ax.axhline(0, color=GRID, lw=1, zorder=1)
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Eval reward")
    ax.set_title("Learning curves, one seed per arm",
                 fontsize=9.5, fontweight="bold", color=INK, loc="left", pad=10)
    ax.margins(x=.16)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=1.4)
    out = os.path.join(RESULTS_DIR, "ablation_curves.png")
    fig.savefig(out, dpi=200, facecolor=SURF)
    print(f"  wrote {out}")


def plot_seeds(seed_rows, order):
    by = {}
    for r in seed_rows:
        by.setdefault(r["name"], []).append(float(r["mean_reward"]))
    names = [n for n in order if n in by]
    if not names:
        return

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    base_vals = by.get("baseline", [])
    if base_vals:
        ax.axvspan(min(base_vals), max(base_vals), color=GRID, zorder=0)
        ax.annotate("baseline spread", (statistics.mean(base_vals), len(names) - .38),
                    ha="center", fontsize=7.5, color=MUTED)

    for i, n in enumerate(names):
        v = by[n]
        yy = len(names) - 1 - i
        ax.plot([min(v), max(v)], [yy, yy], color=GRID, lw=3,
                solid_capstyle="round", zorder=1)
        ax.scatter(v, [yy] * len(v), s=64,
                   color=BASECOL if n == "baseline" else ACCENT,
                   ec=SURF, lw=1.5, zorder=3)
        m = statistics.mean(v)
        ax.scatter([m], [yy], marker="|", s=300, color=INK, lw=2, zorder=4)
        ax.annotate(f"n={len(v)}", (m, yy + .22), ha="center",
                    fontsize=7.5, color=INK)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABEL.get(n, n) for n in reversed(names)],
                       fontsize=8.5, color=INK)
    ax.set_xlabel("Task reward, each dot one seed")
    ax.set_title("An arm whose spread overlaps the baseline band is not "
                 "distinguishable from it",
                 fontsize=9.5, fontweight="bold", color=INK, loc="left", pad=10)
    _style(ax)
    fig.tight_layout(pad=1.4)
    out = os.path.join(RESULTS_DIR, "ablation_seeds.png")
    fig.savefig(out, dpi=200, facecolor=SURF)
    print(f"  wrote {out}")


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": GRID, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "figure.facecolor": SURF, "axes.facecolor": SURF,
    })

    csv_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
    if not os.path.exists(csv_path):
        raise SystemExit("No results yet -- run part3/run_ablation.py first.")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    plot_summary(rows)
    plot_curves([r["name"] for r in rows])

    seed_path = os.path.join(RESULTS_DIR, "seed_check.csv")
    if os.path.exists(seed_path):
        with open(seed_path, newline="", encoding="utf-8") as f:
            plot_seeds(list(csv.DictReader(f)), [r["name"] for r in rows])
    else:
        print("  (no seed_check.csv yet -- skipping the seed figure)")

    txt = os.path.join(RESULTS_DIR, "ablation_summary.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(f"{'config':15}{'reward':>10}{'ep_len':>9}{'kills':>8}{'spawners':>10}\n")
        f.write("-" * 52 + "\n")
        for r in rows:
            f.write(f"{r['name']:15}{float(r['mean_reward']):>10.2f}"
                    f"{float(r['mean_ep_len']):>9.0f}{float(r['mean_kills']):>8.1f}"
                    f"{float(r['mean_spawners']):>10.1f}\n")
    print(f"  wrote {txt}")


if __name__ == "__main__":
    main()
