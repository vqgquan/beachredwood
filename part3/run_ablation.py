"""
part3/run_ablation.py

Trains the baseline and each single-factor variant, then scores them on a
common scale and writes results/ablation_results.csv.

    python part3/run_ablation.py --timesteps 700000 --n-envs 8
    python part3/run_ablation.py --only no_step_cost      # re-run one arm

Design notes that matter for reading the numbers:

  BEST CHECKPOINT, not the final one. PPO on this arena can plateau, break
  through, and regress again inside one run; scoring only the last step risks
  grading an unlucky snapshot. An EvalCallback tracks the best policy during
  training and that is what gets scored.

  COMMON REWARD SCALE. Reward ablations train on their ablated reward but are
  scored on the untouched ArenaEnv. Without this, `no_step_cost` would be
  graded on a scale that never charges it for time and would win for free.

  BEHAVIOUR, NOT JUST REWARD. Episode length, kills and spawners destroyed are
  recorded alongside reward, because two agents can score similarly for
  opposite reasons -- one fighting well, one refusing to fight at all.
"""

import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "part2"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from arena_env import ArenaEnv
from train import linear_schedule            # reuse part2's schedule, do not restate it
from ablation_configs import ABLATION_CONFIGS, EVAL_ON_BASELINE_ENV
from ablation_envs import ABLATION_ENVS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ENV_CLASSES = dict(ABLATION_ENVS, ArenaEnv=ArenaEnv)


def build_ppo(env, seed):
    """PPO with exactly the hyperparameters part2/train.py ships, so the
    ablation baseline is the shipped configuration and not a lookalike."""
    return PPO(
        "MlpPolicy", env,
        learning_rate=linear_schedule(3e-4),
        n_steps=1024, batch_size=256, n_epochs=10,
        gamma=0.995, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        seed=seed, verbose=0,
    )


def score(model, env_cls, scheme, episodes, seed):
    """Run greedy episodes and record behaviour, not only reward."""
    env = env_cls(scheme=scheme)
    rewards, lengths, kills, spawners = [], [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + 5000 + i)
        total, steps, info = 0.0, 0, {}
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            total += r
            steps += 1
            if term or trunc:
                break
        rewards.append(total)
        lengths.append(steps)
        kills.append(info.get("kills", 0))
        spawners.append(info.get("spawners_destroyed", 0))
    env.close()
    return rewards, lengths, kills, spawners


def run_one(cfg, timesteps, n_envs, episodes, seed):
    name, scheme = cfg["name"], cfg["scheme"]
    train_cls = ENV_CLASSES[cfg["env"]]
    eval_cls = ArenaEnv if name in EVAL_ON_BASELINE_ENV else train_cls

    print(f"\n=== [{name}] {cfg['axis']} · train on {cfg['env']} · "
          f"score on {eval_cls.__name__} ===", flush=True)

    train_env = make_vec_env(train_cls, n_envs=n_envs, seed=seed,
                             env_kwargs={"scheme": scheme})
    model = build_ppo(train_env, seed)

    best_dir = os.path.join(RESULTS_DIR, "models", name)
    os.makedirs(best_dir, exist_ok=True)
    # Evaluation is expensive here: a policy that survives reaches the step cap,
    # so one eval episode costs about as much as 1200 training steps. Evaluating
    # every 50k steps with 3 episodes keeps best-checkpoint tracking while
    # holding the overhead to a few percent instead of tripling the run.
    cb = EvalCallback(Monitor(eval_cls(scheme=scheme)),
                      best_model_save_path=best_dir,
                      log_path=os.path.join(RESULTS_DIR, "logs", name),
                      eval_freq=max(50_000 // n_envs, 1),
                      n_eval_episodes=3, deterministic=True, verbose=0)

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)
    train_time = time.time() - t0
    train_env.close()

    best = os.path.join(best_dir, "best_model.zip")
    scored = PPO.load(best) if os.path.exists(best) else model

    r, L, k, s = score(scored, eval_cls, scheme, episodes, seed)
    row = {
        "name": name, "axis": cfg["axis"], "env": cfg["env"],
        "timesteps": timesteps, "seed": seed,
        "mean_reward": round(statistics.mean(r), 2),
        "std_reward": round(statistics.pstdev(r), 2),
        "mean_ep_len": round(statistics.mean(L), 1),
        "mean_kills": round(statistics.mean(k), 2),
        "mean_spawners": round(statistics.mean(s), 2),
        "train_time_sec": round(train_time, 1),
    }
    print(f"    -> reward {row['mean_reward']:+.2f}  len {row['mean_ep_len']:.0f}  "
          f"kills {row['mean_kills']:.1f}  spawners {row['mean_spawners']:.1f}  "
          f"({train_time/60:.1f} min)", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=700_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
    rows = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = {r["name"]: r for r in csv.DictReader(f)}

    todo = [c for c in ABLATION_CONFIGS
            if args.only is None or c["name"] == args.only]
    if not todo:
        raise SystemExit(f"no config named {args.only!r}")

    for cfg in todo:
        rows[cfg["name"]] = run_one(cfg, args.timesteps, args.n_envs,
                                    args.episodes, args.seed)
        ordered = [rows[c["name"]] for c in ABLATION_CONFIGS if c["name"] in rows]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ordered[0].keys()))
            w.writeheader(); w.writerows(ordered)

    print(f"\n[ablation] wrote {csv_path}\n")
    ordered = [rows[c["name"]] for c in ABLATION_CONFIGS if c["name"] in rows]
    base = rows.get("baseline")
    print(f"{'config':15} {'reward':>16} {'ep_len':>8} {'kills':>7} {'spawn':>7}  vs base")
    print("-" * 72)
    for r in ordered:
        d = ("" if not base or r["name"] == "baseline"
             else f"{float(r['mean_reward']) - float(base['mean_reward']):+.2f}")
        print(f"{r['name']:15} {float(r['mean_reward']):+8.2f} "
              f"+/-{float(r['std_reward']):5.2f} {float(r['mean_ep_len']):8.0f} "
              f"{float(r['mean_kills']):7.1f} {float(r['mean_spawners']):7.1f}  {d}")


if __name__ == "__main__":
    main()
