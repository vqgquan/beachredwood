"""
part2/capture_screenshots.py

Renders the arena headlessly and saves PNG stills for the report and for the
`screenshots/` folder the submission checklist asks for. No visible window is
needed, so this runs anywhere -- including a machine with no display.

    python capture_screenshots.py --model models/best/direct/best_model.zip

Saves into ../screenshots/:
    arena_combat.png    mid-fight: player, enemies, spawners, projectiles
    arena_phase.png     the frame right after a phase advance
    arena_early.png     opening state, spawners intact

If no model is given the agent acts randomly, which is still enough to produce
a populated scene, but a trained agent gives a better-looking screenshot
because it actually engages the spawners.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from arena_env import ArenaEnv

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "screenshots"
)


def save(surface, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    pygame.image.save(surface, path)
    print(f"  saved {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="path to a trained .zip")
    ap.add_argument("--scheme", default="direct", choices=["direct", "rotate"])
    ap.add_argument("--max-steps", type=int, default=3000)
    args = ap.parse_args()

    policy = None
    if args.model and os.path.exists(args.model):
        from stable_baselines3 import PPO
        policy = PPO.load(args.model)
        print(f"[capture] using {args.model}")
    else:
        print("[capture] no model given -- random actions")

    env = ArenaEnv(control_scheme=args.scheme, render_mode="human")
    obs, _ = env.reset(seed=7)

    shots = {}
    phase_seen = None

    for step in range(args.max_steps):
        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = env.action_space.sample()

        obs, _, terminated, truncated, _ = env.step(action)
        env.render()
        surface = pygame.display.get_surface()
        if surface is None:
            raise SystemExit("[capture] no surface -- does render() open a display?")

        state = env.sim.get_state_dict()

        if step == 40 and "early" not in shots:
            shots["early"] = save(surface.copy(), "arena_early.png")

        # a frame with enemies AND at least one bullet in flight reads best
        if ("combat" not in shots and step > 60
                and state.get("n_enemies", 0) >= 2):
            shots["combat"] = save(surface.copy(), "arena_combat.png")

        if phase_seen is None:
            phase_seen = state.get("phase_index", 0)
        elif state.get("phase_index", 0) > phase_seen and "phase" not in shots:
            shots["phase"] = save(surface.copy(), "arena_phase.png")
            phase_seen = state.get("phase_index", 0)

        if len(shots) == 3:
            break
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    missing = {"early", "combat", "phase"} - set(shots)
    if missing:
        print(f"[capture] not captured: {sorted(missing)} "
              f"-- run again with a longer --max-steps or a better model")
    print(f"[capture] {len(shots)}/3 screenshots in {OUT_DIR}")


if __name__ == "__main__":
    main()
