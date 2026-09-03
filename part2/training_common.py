"""
training_common.py
===================
Shared helpers for the two training scripts (train_control1.py /
train_control2.py) and the Part III ablation runner. Keeps PPO
hyperparameters and vec-env construction in one place so both control
schemes are trained identically apart from the control_scheme flag itself
(a fair comparison, and less code to keep in sync).

IMPORTANT: this module sets SDL_VIDEODRIVER=dummy at import time so
Pygame never opens a real window during training, even if some
dependency imports pygame internally. Eval scripts unset this before
importing.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless training, per architecture doc

from typing import Callable, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from arena_env import ArenaEnv

SEED = 42  # fixed seed for reproducibility, per architecture doc


def make_env_fn(control_scheme: str,
                 shaping_enabled: bool = True,
                 include_spawner_feature: bool = True,
                 seed: int = SEED) -> Callable[[], ArenaEnv]:
    """Returns a thunk building one Monitor-wrapped ArenaEnv instance.
    make_vec_env calls this once per parallel worker."""
    def _init():
        env = ArenaEnv(
            control_scheme=control_scheme,
            shaping_enabled=shaping_enabled,
            include_spawner_feature=include_spawner_feature,
            seed=seed,
        )
        return Monitor(env)
    return _init


def build_vec_envs(control_scheme: str, n_envs: int = 8,
                    shaping_enabled: bool = True,
                    include_spawner_feature: bool = True,
                    eval_shaping_enabled: bool = False,
                    seed: int = SEED):
    """Vectorized training env + a separate single-env for EvalCallback
    (SB3 best practice: eval env should not share state with training envs).

    IMPORTANT (fixed 2026-08-27): the eval env defaults to
    shaping_enabled=False regardless of the training env's setting. The
    potential-based shaping term is deliberately part of the *training*
    signal (it helps PPO find spawners faster), but if EvalCallback's
    reported "mean_reward" also includes shaping, that number stops
    meaning "how good is this policy at the task" -- it becomes "how good
    is this policy at staying close to spawners AND doing the task",
    which can look far better than true task performance (verified: a
    500k-step 'direct' model reported ~+4 eval reward WITH shaping still
    on, but only ~-23 when evaluated on task reward alone). Evaluating
    (and therefore EvalCallback's best-model selection) on task-only
    reward is what makes "best_model.zip" actually mean the best model at
    the task, and keeps this consistent with part3_ablation/run_ablation.py,
    which already evaluates every ablation config with shaping off for the
    same reason."""
    train_env = make_vec_env(
        make_env_fn(control_scheme, shaping_enabled, include_spawner_feature, seed),
        n_envs=n_envs,
        seed=seed,
    )
    eval_env = make_vec_env(
        make_env_fn(control_scheme, eval_shaping_enabled, include_spawner_feature, seed + 1000),
        n_envs=1,
        seed=seed + 1000,
    )
    return train_env, eval_env


def _tensorboard_available() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


def build_ppo(env, net_arch=(64, 64), learning_rate: float = 3e-4,
              tensorboard_log: Optional[str] = None, seed: int = SEED) -> PPO:
    """PPO(MlpPolicy) with the architecture doc's stated rationale:
    on-policy, fewer hyperparameter traps than DQN for this obs space.
    net_arch / learning_rate are exposed as args because Part III's
    ablation study sweeps exactly these two."""
    policy_kwargs = dict(net_arch=dict(pi=list(net_arch), vf=list(net_arch)))
    if tensorboard_log is not None and not _tensorboard_available():
        print("[training_common] tensorboard not installed -- disabling TB logging "
              "(pip install tensorboard to enable). Training continues normally.")
        tensorboard_log = None
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        seed=seed,
        verbose=1,
    )
    return model


def build_callbacks(control_scheme: str, save_freq: int = 20_000,
                     eval_env=None, eval_freq: int = 10_000,
                     models_dir: str = "models"):
    ckpt_dir = os.path.join(models_dir, "checkpoints", control_scheme)
    best_dir = os.path.join(models_dir, "best", control_scheme)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(save_freq // 8, 1),  # save_freq counted per env in SB3 vec envs
        save_path=ckpt_dir,
        name_prefix=f"ppo_{control_scheme}",
    )
    callbacks = [checkpoint_cb]
    if eval_env is not None:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=best_dir,
            log_path=os.path.join("logs", control_scheme),
            eval_freq=max(eval_freq // 8, 1),
            n_eval_episodes=10,
            deterministic=True,
        )
        callbacks.append(eval_cb)
    return callbacks
