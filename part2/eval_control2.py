"""
eval_control2.py
=================
Loads the trained "direct" model and runs it with rendering ON, for
qualitative inspection and for recording the demo video. This is the ONLY
place render=True is used -- training stays headless per architecture doc.

Usage:
    python eval_control2.py [--model models/best/direct/best_model.zip]
                             [--episodes 5] [--fps 30]

IMPORTANT for the demo video: make sure the model path here is the exact
file being submitted (a common way to lose marks per the risk list is
"model trong video khác model nộp").
"""
import argparse
import os
import time

# Respect any SDL_VIDEODRIVER already set (e.g. dummy for headless eval).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from stable_baselines3 import PPO
from arena_env import ArenaEnv

CONTROL_SCHEME = "direct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=f"models/best/{CONTROL_SCHEME}/best_model.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        fallback = f"models/ppo_{CONTROL_SCHEME}.zip"
        print(f"[eval_control2] {args.model} not found, falling back to {fallback}")
        args.model = fallback

    print(f"[eval_control2] loading {args.model}")
    model = PPO.load(args.model)

    env = ArenaEnv(control_scheme=CONTROL_SCHEME, render_mode="human")
    for ep in range(args.episodes):
        obs, info = env.reset(seed=1000 + ep)
        env.render()
        terminated = truncated = False
        ep_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            env.render()
            time.sleep(1.0 / args.fps)
        outcome = "WIN" if info.get("episode_won") else ("TIMEOUT" if truncated else "DIED")
        print(f"[eval_control2] episode {ep + 1}/{args.episodes}: "
              f"reward={ep_reward:.1f} outcome={outcome} "
              f"final_phase={info.get('phase_index')}")
    env.close()


if __name__ == "__main__":
    main()
