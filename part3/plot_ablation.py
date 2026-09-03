"""
part3/plot_ablation.py
=================================
Reads results/ablation_results.csv (produced by run_ablation.py) and
renders one bar chart per axis: baseline vs. that axis's variant, with
mean_reward and std_reward as an error bar. Saves PNGs for the report.

Usage:
    python part3/plot_ablation.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CSV_PATH = os.path.join(RESULTS_DIR, "ablation_results.csv")

AXIS_TITLES = {
    "shaping": "Potential-based reward shaping (on vs off)",
    "observation": "Observation completeness (full vs no-spawner-feature)",
    "network": "Network capacity ([64,64] vs [128,128])",
}


def load_rows():
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def plot_axis(rows, axis, baseline_row, out_dir):
    variant_rows = [r for r in rows if r["axis"] == axis]
    if not variant_rows:
        return
    pair = [baseline_row] + variant_rows
    names = [r["name"] for r in pair]
    means = [float(r["mean_reward"]) for r in pair]
    stds = [float(r["std_reward"]) for r in pair]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["#4C72B0"] + ["#DD8452"] * len(variant_rows)
    ax.bar(names, means, yerr=stds, capsize=6, color=colors)
    ax.set_ylabel("Mean episode reward (eval, shaping OFF for fair scoring)")
    ax.set_title(AXIS_TITLES.get(axis, axis))
    ax.axhline(0, color="gray", linewidth=0.8)
    plt.xticks(rotation=15)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"ablation_{axis}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_ablation] wrote {out_path}")


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"No results yet -- run part3/run_ablation.py first "
                          f"(expected {CSV_PATH})")
    rows = load_rows()
    baseline_row = next(r for r in rows if r["axis"] == "baseline")
    axes_present = sorted({r["axis"] for r in rows if r["axis"] != "baseline"})
    for axis in axes_present:
        plot_axis(rows, axis, baseline_row, RESULTS_DIR)

    # combined summary table as a text file for quick reference in the report
    summary_path = os.path.join(RESULTS_DIR, "ablation_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"{'config':22s} {'axis':12s} {'mean_reward':>12s} {'std':>8s} {'train_s':>9s}\n")
        for r in rows:
            f.write(f"{r['name']:22s} {r['axis']:12s} "
                     f"{float(r['mean_reward']):12.2f} {float(r['std_reward']):8.2f} "
                     f"{r['train_time_sec']:>9s}\n")
    print(f"[plot_ablation] wrote {summary_path}")


if __name__ == "__main__":
    main()
