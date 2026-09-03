"""
train_control1.py
==================
Trains PPO on the "rotate" control scheme (Discrete(5): noop, turn_left,
turn_right, thrust, fire -- Asteroids-style momentum steering).

Usage:
    python train_control1.py [--timesteps 500000] [--n-envs 8]

Output: models/ppo_rotate.zip  (final model)
        models/best/rotate/best_model.zip  (best model per EvalCallback)
        models/checkpoints/rotate/  (periodic checkpoints)
        logs/rotate/  (EvalCallback eval logs, npz)
"""
import argparse
import os
import time

from training_common import build_vec_envs, build_ppo, build_callbacks, SEED

CONTROL_SCHEME = "rotate"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[64, 64])
    args = parser.parse_args()

    train_env, eval_env = build_vec_envs(CONTROL_SCHEME, n_envs=args.n_envs, seed=SEED)
    model = build_ppo(train_env, net_arch=tuple(args.net_arch), learning_rate=args.lr,
                       tensorboard_log="logs/tensorboard", seed=SEED)
    callbacks = build_callbacks(CONTROL_SCHEME, eval_env=eval_env)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                tb_log_name=CONTROL_SCHEME, progress_bar=True)
    print(f"[train_control1] finished {args.timesteps} steps in {time.time() - t0:.1f}s")

    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_rotate.zip")
    print("[train_control1] saved final model to models/ppo_rotate.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
