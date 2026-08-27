"""
play_human.py
Play the arena yourself with the keyboard (useful to check the game works).

    python play_human.py --scheme rotation
    python play_human.py --scheme direct

rotation: UP = thrust, LEFT/RIGHT = rotate, SPACE = shoot
direct:   ARROW KEYS = move, SPACE = shoot
"""

import argparse
import pygame
from arena_env import ArenaEnv


def get_action(keys, scheme):
    """Translate the pressed keys into one discrete action."""
    if scheme == "rotation":
        if keys[pygame.K_SPACE]:
            return 4
        if keys[pygame.K_UP]:
            return 1
        if keys[pygame.K_LEFT]:
            return 2
        if keys[pygame.K_RIGHT]:
            return 3
    else:
        if keys[pygame.K_SPACE]:
            return 5
        if keys[pygame.K_UP]:
            return 1
        if keys[pygame.K_DOWN]:
            return 2
        if keys[pygame.K_LEFT]:
            return 3
        if keys[pygame.K_RIGHT]:
            return 4
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", choices=["rotation", "direct"], default="rotation")
    args = p.parse_args()

    env = ArenaEnv(scheme=args.scheme, render_mode="human")
    obs, _ = env.reset(seed=0)
    env.render()          # opens the window before we start reading the keyboard
    done, total = False, 0.0
    while not done:
        pygame.event.pump()               # refresh the keyboard state
        keys = pygame.key.get_pressed()
        obs, r, terminated, truncated, info = env.step(get_action(keys, args.scheme))
        total += r
        done = terminated or truncated or env.screen is None   # window closed
    print(f"Reward {total:.2f} | phase {info['phase']} | kills {info['kills']}")
    env.close()


if __name__ == "__main__":
    main()