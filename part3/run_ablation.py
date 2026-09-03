"""
part3/run_ablation.py
===============================
Part III (Masters, Option A) ablation study runner.

Trains the BASELINE config and each single-factor VARIANT (see
ablation_configs.py) for the same timestep budget, evaluates each with
`evaluate_policy` over a held-out eval env, and writes one row per config
to results/ablation_results.csv. Run plot_ablation.py afterwards to turn
that CSV into comparison charts (PNG) for the report.

Usage:
    python part3/run_ablation.py --timesteps 150000 --n-envs 8

Notes:
- Each config gets its own fresh seeded envs (see training_common.SEED),
  so results are comparable across configs.
- timesteps defaults lower than the "real" Part II training runs because
  this script trains 4 models back-to-back; raise --timesteps once you've
  confirmed the pipeline works end-to-end on your machine.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "part2"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

from training_common import build_vec_envs, build_ppo, SEED
from ablation_configs import ABLATION_CONFIGS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def run_one_config(cfg: dict, timesteps: int, n_envs: int, eval_episodes: int, eval_freq: int = 20_000) -> dict:
    print(f"\n=== [{cfg['name']}] axis={cfg['axis']} "
          f"shaping={cfg['shaping_enabled']} spawner_feat={cfg['include_spawner_feature']} "
          f"net_arch={cfg['net_arch']} lr={cfg['lr']} ===")

    # eval_env defaults to shaping_enabled=False (see training_common.build_vec_envs
    # docstring, fixed 2026-08-27) so EvalCallback tracks TASK-only reward, not a
    # shaping-inflated number.
    train_env, eval_env = build_vec_envs(
        cfg["control_scheme"], n_envs=n_envs,
        shaping_enabled=cfg["shaping_enabled"],
        include_spawner_feature=cfg["include_spawner_feature"],
        seed=SEED,
    )
    model = build_ppo(train_env, net_arch=cfg["net_arch"], learning_rate=cfg["lr"], seed=SEED)

    # Track the BEST checkpoint during training, not just whatever the policy
    # happens to be at the final step. Empirically (see arena_rl training log,
    # 2026-08-27) PPO on this task can plateau for hundreds of thousands of
    # steps, break through to a much better policy, and later regress/collapse
    # again before total_timesteps is reached -- evaluating only the final
    # model (the previous version of this script) risks scoring an unlucky
    # collapsed snapshot instead of the config's actual best achievable
    # performance, which would bias the ablation's conclusions.
    best_dir = os.path.join(RESULTS_DIR, "models", cfg["name"], "best")
    os.makedirs(best_dir, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=os.path.join(RESULTS_DIR, "logs", cfg["name"]),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)
    train_time = time.time() - t0

    best_model_path = os.path.join(best_dir, "best_model.zip")
    eval_model = PPO.load(best_model_path) if os.path.exists(best_model_path) else model

    # Final scoring pass: fresh held-out episodes (different seed from the
    # EvalCallback's own eval_env draws) with shaping OFF, so every config
    # -- ablated or not -- is scored on the same task-reward scale.
    fair_eval_env, _ = build_vec_envs(
        cfg["control_scheme"], n_envs=1,
        shaping_enabled=False,
        include_spawner_feature=cfg["include_spawner_feature"],
        seed=SEED + 2000,
    )
    mean_reward, std_reward = evaluate_policy(
        eval_model, fair_eval_env, n_eval_episodes=eval_episodes, deterministic=True
    )

    row = {
        "name": cfg["name"],
        "axis": cfg["axis"],
        "shaping_enabled": cfg["shaping_enabled"],
        "include_spawner_feature": cfg["include_spawner_feature"],
        "net_arch": "x".join(map(str, cfg["net_arch"])),
        "lr": cfg["lr"],
        "timesteps": timesteps,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "train_time_sec": round(train_time, 1),
        "used_best_checkpoint": os.path.exists(best_model_path),
    }

    final_model_path = os.path.join(RESULTS_DIR, "models", cfg["name"], "final.zip")
    model.save(final_model_path)
    if os.path.exists(best_model_path):
        eval_model.save(os.path.join(RESULTS_DIR, "models", f"ppo_{cfg['name']}.zip"))
    else:
        model.save(os.path.join(RESULTS_DIR, "models", f"ppo_{cfg['name']}.zip"))

    train_env.close()
    eval_env.close()
    fair_eval_env.close()
    return row


def _load_existing_rows(csv_path: str) -> dict:
    """Load prior results (keyed by config name) so re-running one config
    with --only doesn't clobber the other configs' already-computed rows."""
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, newline="") as f:
        return {r["name"]: r for r in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=150_000,
                         help="timesteps PER config")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--only", type=str, default=None,
                         help="Run a single config by name (e.g. 'baseline') instead of "
                              "all 4 -- useful to split a long run across multiple sessions "
                              "when wall-clock per call is limited. Results merge into the "
                              "existing CSV rather than overwriting other configs' rows.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "ablation_results.csv")

    configs_to_run = ABLATION_CONFIGS
    if args.only is not None:
        configs_to_run = [c for c in ABLATION_CONFIGS if c["name"] == args.only]
        if not configs_to_run:
            valid = [c["name"] for c in ABLATION_CONFIGS]
            raise SystemExit(f"Unknown config '{args.only}'. Valid names: {valid}")

    existing = _load_existing_rows(csv_path)
    for cfg in configs_to_run:
        row = run_one_config(cfg, args.timesteps, args.n_envs, args.eval_episodes)
        existing[cfg["name"]] = {k: str(v) for k, v in row.items()}
        print(f"[{cfg['name']}] mean_reward={row['mean_reward']:.2f} "
              f"+/- {row['std_reward']:.2f}  best_checkpoint_used={row['used_best_checkpoint']}  "
              f"({row['train_time_sec']}s train)")

    # Preserve ABLATION_CONFIGS order in the output CSV regardless of which
    # subset was just (re-)run.
    ordered_rows = [existing[c["name"]] for c in ABLATION_CONFIGS if c["name"] in existing]
    if ordered_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(ordered_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ordered_rows)
        print(f"\n[run_ablation] wrote results to {csv_path} ({len(ordered_rows)}/{len(ABLATION_CONFIGS)} configs present)")
        if len(ordered_rows) < len(ABLATION_CONFIGS):
            missing = [c["name"] for c in ABLATION_CONFIGS if c["name"] not in existing]
            print(f"[run_ablation] still missing: {missing} -- run with --only <name> for each remaining config")
        else:
            print("[run_ablation] next: python part3/plot_ablation.py")


if __name__ == "__main__":
    main()
