"""
part3/demo_ablation.py
Play a trained ablation model in the env it was actually trained on.

evaluate.py always builds a plain ArenaEnv. That is fine for the baseline,
but NoShipFrame was trained with observation dims 9,10,15,16 held at zero --
running it on ArenaEnv feeds it inputs it never saw, which changes its
behaviour (it dies early instead of running away). This script pairs each
model with its own env class so the demo shows what the study measured.

    python part3/demo_ablation.py baseline
    python part3/demo_ablation.py no_ship_frame
    python part3/demo_ablation.py no_shaping --episodes 3
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "part2"))

from stable_baselines3 import PPO  # noqa: E402

from arena_env import ArenaEnv  # noqa: E402
from ablation_envs import ABLATION_ENVS  # noqa: E402
from ablation_configs import ABLATION_CONFIGS, EVAL_ON_BASELINE_ENV  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ENVS = dict(ABLATION_ENVS, ArenaEnv=ArenaEnv)
CFG_BY_NAME = {c["name"]: c for c in ABLATION_CONFIGS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=sorted(CFG_BY_NAME),
                    help="which ablation arm to demo")
    ap.add_argument("--episodes", type=int, default=2)
    args = ap.parse_args()

    cfg = CFG_BY_NAME[args.arm]
    model_path = os.path.join(RESULTS_DIR, "models", args.arm, "best_model.zip")
    model = PPO.load(model_path)
    # Same scoring rule as the study: reward ablations are graded on the
    # untouched ArenaEnv, the observation ablation on its own interface.
    eval_cls = ArenaEnv if args.arm in EVAL_ON_BASELINE_ENV else ENVS[cfg["env"]]
    env = eval_cls(scheme=cfg["scheme"], render_mode="human")
    print(f"[demo] {args.arm}: model trained on {cfg['env']}, "
          f"shown on {eval_cls.__name__} ({cfg['scheme']})")

    rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=100 + ep)
        total, steps, done = 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            total += r
            steps += 1
            done = term or trunc
            env.render()
        rewards.append(total)
        print(f"Episode {ep + 1}: reward {total:+8.2f} | steps {steps:4d} | "
              f"kills {info.get('kills', 0)} | "
              f"spawners {info.get('spawners_destroyed', 0)}")
    env.close()
    print(f"[demo] mean reward {sum(rewards) / len(rewards):+.2f}")


if __name__ == "__main__":
    main()
