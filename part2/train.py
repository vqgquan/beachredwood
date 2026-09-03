"""
train.py
Trains one agent (PPO or DQN) on one control scheme, headless (no window).

Examples:
    python train.py --scheme rotation --algo ppo --timesteps 700000
    python train.py --scheme direct   --algo ppo --timesteps 700000
    python train.py --scheme rotation --n-envs 16 --subproc     (fastest for PPO)
    python train.py --scheme rotation --algo dqn --device cuda  (GPU helps DQN)

Models go to models/, TensorBoard logs to tensorboard/.
"""

import argparse
import os

import torch

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from arena_env import ArenaEnv

MODEL_DIR = "models"
LOG_DIR = "tensorboard"


def linear_schedule(start):
    """Learning rate decays from `start` to 0 over training (helps late-stage stability)."""
    return lambda progress_remaining: progress_remaining * start


def pick_device(choice):
    """Resolve the --device flag and warn when a GPU would not actually help."""
    if choice == "auto":
        return "auto"
    if choice == "cuda" and not torch.cuda.is_available():
        print("CUDA not available (CPU-only torch build?) -> falling back to CPU")
        return "cpu"
    return choice


def build_model(algo, scheme, n_envs, seed, device="auto", subproc=False):
    """Create the agent with tuned hyperparameters (see README for the sweep)."""
    if algo == "ppo":
        # SubprocVecEnv runs each env in its own process = uses all CPU cores.
        # This is the real speedup for PPO here, because the pygame loop is the bottleneck.
        env = make_vec_env(ArenaEnv, n_envs=n_envs, seed=seed,
                           env_kwargs={"scheme": scheme},
                           vec_env_cls=SubprocVecEnv if subproc else None)
        model = PPO(
            "MlpPolicy", env,
            learning_rate=linear_schedule(3e-4),
            n_steps=1024,            # 1024 x 8 envs = 8192 samples per update
            batch_size=256,
            n_epochs=10,
            gamma=0.995,             # long episodes (2500 steps) need a high gamma
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,           # keeps exploring the rarely used "shoot" action
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
            tensorboard_log=LOG_DIR,
            device=device,
            seed=seed,
            verbose=1,
        )
    else:
        env = Monitor(ArenaEnv(scheme=scheme))   # DQN uses a single environment
        model = DQN(
            "MlpPolicy", env,
            learning_rate=5e-4,
            buffer_size=200_000,
            learning_starts=10_000,
            batch_size=128,
            tau=1.0,
            gamma=0.995,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=2000,
            exploration_fraction=0.3,     # slow epsilon decay: the arena needs exploration
            exploration_final_eps=0.05,
            policy_kwargs=dict(net_arch=[128, 128]),
            tensorboard_log=LOG_DIR,
            device=device,
            seed=seed,
            verbose=1,
        )
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", choices=["rotation", "direct"], default="rotation")
    p.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    p.add_argument("--timesteps", type=int, default=700_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="cuda helps DQN; PPO with a small MLP is usually faster on cpu")
    p.add_argument("--subproc", action="store_true",
                   help="run the envs in separate processes (recommended with --n-envs 12+)")
    args = p.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = pick_device(args.device)
    name = f"{args.algo}_{args.scheme}"
    model = build_model(args.algo, args.scheme, args.n_envs, args.seed, device, args.subproc)
    print(f"Training {name} on device: {model.device}")

    # save a checkpoint every 50k steps in case training is interrupted
    ckpt = CheckpointCallback(save_freq=max(50_000 // args.n_envs, 1),
                              save_path=os.path.join(MODEL_DIR, "checkpoints"),
                              name_prefix=name)

    model.learn(total_timesteps=args.timesteps, callback=ckpt, tb_log_name=name)
    model.save(os.path.join(MODEL_DIR, name))
    print(f"Saved {MODEL_DIR}/{name}.zip")


if __name__ == "__main__":
    # the __main__ guard is required for --subproc on Windows
    main()
