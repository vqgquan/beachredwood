#!/usr/bin/env python3
"""Interactive Pygame gridworld — Part I graded deliverable.

Trains Q-learning / SARSA live, on screen, with keyboard control over level,
algorithm, intrinsic reward (Level 6), and playback speed. Run with:

    python app.py

Controls: 1-7 pick a level, Q/S pick the algorithm, I toggles the Level 6
intrinsic reward, T toggles rapid (unrendered) training speed, V toggles
per-step rendering, P pauses evaluation playback, R soft-resets the current
run (keeps the episode counter), Esc quits.
"""
import pygame

from agents import (QTable, epsilon_greedy, greedy_action, linear_epsilon,
                     q_learning_update, sarsa_update)
from config import (EPS_DECAY_EP, EPS_END, EPS_START, EPISODES, FPS_FAST,
                     FPS_VISUAL, INTRINSIC_REWARD_STRENGTH, MAX_STEPS,
                     PANEL_W, RAPID_RENDER_EVERY, RAPID_STEPS, STEP_PENALTY,
                     TILE_SIZE, ALPHA, GAMMA)
from env import GridWorld
from levels import LEVELS, LEVEL_LABELS
from metrics import make_metrics, record_episode, save_training_curve

import math

# -----------------------------
# Pygame window
# -----------------------------
GRID_W, GRID_H = 12, 8
GAME_W, GAME_H = GRID_W * TILE_SIZE, GRID_H * TILE_SIZE
WIDTH, HEIGHT = GAME_W + PANEL_W, GAME_H

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("GridWorld - Part I Q-Learning & SARSA")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)
font_small = pygame.font.SysFont("consolas", 16)

# Colors
COL_BG = (22, 24, 30)
COL_PANEL = (16, 18, 22)
COL_GRID = (50, 54, 64)
COL_AGENT = (74, 222, 128)
COL_APPLE = (252, 92, 101)
COL_KEY = (247, 210, 63)
COL_CHEST = (180, 120, 80)
COL_ROCK = (90, 95, 110)
COL_FIRE = (240, 90, 40)
COL_MONSTER = (170, 70, 210)
COL_TEXT = (240, 240, 240)
COL_MUTED = (170, 175, 185)
COL_ACCENT = (90, 170, 255)


# -----------------------------
# Drawing
# -----------------------------
def draw_text(surface, text, x, y, color=COL_TEXT, center=False, small=False):
    img = (font_small if small else font).render(text, True, color)
    rect = img.get_rect()
    if center:
        rect.center = (x, y)
    else:
        rect.topleft = (x, y)
    surface.blit(img, rect)


def draw_panel(surface, mode_level, algo_name, ep, step, eps, total_reward,
               env, qtab, fast_mode, comparison=None):
    panel_x = GAME_W
    pygame.draw.rect(surface, COL_PANEL, pygame.Rect(panel_x, 0, PANEL_W, HEIGHT))
    pygame.draw.line(surface, COL_GRID, (panel_x, 0), (panel_x, HEIGHT), 2)

    x = panel_x + 12
    y = 8
    def line(text, color=COL_TEXT, gap=16, heading=False):
        nonlocal y
        draw_text(surface, text, x, y, color, small=not heading)
        y += gap

    line("Assignment 3 - Part I", COL_TEXT, 20, True)
    line(LEVEL_LABELS.get(mode_level, f"Level {mode_level}"), COL_ACCENT, 18, False)
    line(f"Mode: {algo_name}", COL_TEXT, 17, False)
    # During evaluation ep == EPISODES, so do not display a fictitious
    # episode 801/800 after the final training episode.
    shown_episode = min(ep + 1, EPISODES)
    line(f"Episode {shown_episode}/{EPISODES}  Step {step}/{MAX_STEPS}", COL_TEXT, 16)
    line(f"Epsilon {eps:.3f}  Return {total_reward:.2f}", COL_TEXT, 16)
    line(f"Intrinsic reward: {'on' if mode_level == 6 else 'off'}", COL_TEXT, 16)
    items = env.remaining_items()
    line(f"Apples {items['apples']}  Keys {items['keys']}  Chests {items['chests_left']}", COL_TEXT)
    line(f"Level {mode_level} comparison", COL_ACCENT, 16, True)
    q_stats = (comparison or {}).get(mode_level, {}).get("q")
    s_stats = (comparison or {}).get(mode_level, {}).get("sarsa")
    def comparison_line(label, values):
        if values is None:
            line(f"{label}: not trained", COL_MUTED)
        else:
            average = values["return_total"] / max(1, values["episodes"])
            successful_count = max(1, values["successes"])
            avg_steps = values["successful_steps"] / successful_count
            deaths = values["deaths"]
            hazards = values["fire_deaths"] + values["monster_deaths"]
            line(f"{label}: R{average:.2f} S{values['successes']}/{values['episodes']}", COL_TEXT)
            line(f"   steps {avg_steps:.1f}  D{deaths} H{hazards} T{values['timeouts']}", COL_MUTED)
    comparison_line("Q", q_stats)
    comparison_line("S", s_stats)
    line("Controls", COL_ACCENT, 16, True)
    line("1-7 level   Q/S algorithm", COL_MUTED)
    line("I intrinsic L6   T rapid", COL_MUTED)
    line("V visual   R reset   P pause", COL_MUTED)
    line("ESC quit", COL_MUTED)
    line(f"States {len(qtab.q)}   Fast {'on' if fast_mode else 'off'}", COL_TEXT)


def draw_world(env: GridWorld):
    for y in range(env.h):
        for x in range(env.w):
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, COL_BG, rect)
            pygame.draw.rect(screen, COL_GRID, rect, 1)
            if (x, y) in env.rocks:
                pygame.draw.rect(screen, COL_ROCK, rect.inflate(-8, -8), border_radius=4)
            if (x, y) in env.fires:
                pygame.draw.rect(screen, COL_FIRE, rect.inflate(-12, -12), border_radius=8)

    for p, idx in env.apple_index.items():
        if (env.apple_mask >> idx) & 1:
            cx, cy = p[0] * TILE_SIZE + TILE_SIZE // 2, p[1] * TILE_SIZE + TILE_SIZE // 2
            pygame.draw.circle(screen, COL_APPLE, (cx, cy), TILE_SIZE // 4)

    for p, idx in env.key_index.items():
        if (env.key_mask >> idx) & 1:
            cx, cy = p[0] * TILE_SIZE + TILE_SIZE // 2, p[1] * TILE_SIZE + TILE_SIZE // 2
            pygame.draw.circle(screen, COL_KEY, (cx, cy), TILE_SIZE // 5)

    for p, idx in env.chest_index.items():
        if idx not in env.opened_chests:
            rect = pygame.Rect(p[0] * TILE_SIZE + 10, p[1] * TILE_SIZE + 12, TILE_SIZE - 20, TILE_SIZE - 22)
            pygame.draw.rect(screen, COL_CHEST, rect, border_radius=4)

    for m in env.monsters:
        rect = pygame.Rect(m[0] * TILE_SIZE + 8, m[1] * TILE_SIZE + 8, TILE_SIZE - 16, TILE_SIZE - 16)
        pygame.draw.rect(screen, COL_MONSTER, rect, border_radius=6)

    ax, ay = env.agent
    pygame.draw.rect(
        screen,
        COL_AGENT,
        pygame.Rect(ax * TILE_SIZE + 8, ay * TILE_SIZE + 8, TILE_SIZE - 16, TILE_SIZE - 16),
        border_radius=6,
    )


def render(env, mode_level, algo_name, ep, step, eps, total_reward,
           qtab, rapid_mode, comparison=None):
    screen.fill(COL_PANEL)
    draw_world(env)
    draw_panel(screen, mode_level, algo_name, ep, step, eps, total_reward,
               env, qtab, rapid_mode, comparison)
    pygame.display.flip()


# -----------------------------
# Level / algorithm switching
# -----------------------------
def fresh_run(level_idx, algo_name):
    """Fresh environment + Q-table for level_idx/algo_name, with the episode
    counter and epsilon schedule reset to the start (level or algorithm switch)."""
    env = GridWorld(LEVELS[level_idx])
    qtab = QTable()
    metrics = make_metrics()
    intrinsic_enabled = level_idx == 6
    s = env.reset()
    eps = EPS_START
    a = epsilon_greedy(qtab, s, eps) if algo_name == "sarsa" else None
    return env, qtab, metrics, intrinsic_enabled, s, eps, a


# -----------------------------
# Main training loop
# -----------------------------
def main():
    current_level = 0
    current_algo = "q"
    env, qtab, metrics, intrinsic_enabled, s, eps, a = fresh_run(current_level, current_algo)
    comparison = {level: {"q": None, "sarsa": None} for level in LEVELS}
    show_visuals = True
    rapid_mode = False
    running = True
    mode = "train"
    ep = 0
    total_reward = 0.0
    intrinsic_episode = 0.0
    env_reward_episode = 0.0
    steps = 0
    pause_eval = False
    eval_step = 0
    eval_initialized = False
    visit_counts = {}

    while running:
        if mode == "train" and ep >= EPISODES:
            comparison[current_level][current_algo] = dict(metrics)
            mode = "eval"
            env = GridWorld(LEVELS[current_level])
            s = env.reset()
            total_reward = 0.0
            steps = 0
            eval_step = 0
            eval_initialized = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                if event.key == pygame.K_v:
                    show_visuals = not show_visuals
                elif event.key == pygame.K_t:
                    rapid_mode = not rapid_mode
                elif event.key == pygame.K_p and mode == "eval":
                    pause_eval = not pause_eval
                elif event.key == pygame.K_r and mode == "train":
                    # Soft reset: wipe the Q-table but keep the episode count
                    # (and therefore the epsilon schedule) progressing.
                    qtab = QTable()
                    metrics = make_metrics()
                    s = env.reset()
                    total_reward = 0.0
                    steps = 0
                    eps = linear_epsilon(ep, EPS_START, EPS_END, EPS_DECAY_EP)
                    if current_algo == "sarsa":
                        a = epsilon_greedy(qtab, s, eps)
                elif pygame.K_1 <= event.key <= pygame.K_7:
                    current_level = event.key - pygame.K_1
                    env, qtab, metrics, intrinsic_enabled, s, eps, a = fresh_run(
                        current_level, current_algo)
                    ep = 0
                    mode = "train"
                    eval_initialized = False
                    total_reward = 0.0
                    intrinsic_episode = 0.0
                    env_reward_episode = 0.0
                    steps = 0
                elif event.key == pygame.K_i and current_level == 6:
                    intrinsic_enabled = not intrinsic_enabled
                    env, qtab, metrics, _, s, eps, a = fresh_run(current_level, current_algo)
                    ep = 0
                    mode = "train"
                    eval_initialized = False
                    total_reward = 0.0
                    intrinsic_episode = 0.0
                    env_reward_episode = 0.0
                    steps = 0
                elif event.key == pygame.K_q:
                    current_algo = "q"
                    env, qtab, metrics, intrinsic_enabled, s, eps, a = fresh_run(
                        current_level, current_algo)
                    ep = 0
                    mode = "train"
                    eval_initialized = False
                    total_reward = 0.0
                    steps = 0
                elif event.key == pygame.K_s:
                    current_algo = "sarsa"
                    env, qtab, metrics, intrinsic_enabled, s, eps, a = fresh_run(
                        current_level, current_algo)
                    ep = 0
                    mode = "train"
                    eval_initialized = False
                    total_reward = 0.0
                    steps = 0

        if not running:
            break

        if mode == "train":
            eps = linear_epsilon(ep, EPS_START, EPS_END, EPS_DECAY_EP)
            # Counts are episode-local, including when rapid mode performs
            # several actions in one rendered frame.
            if steps == 0:
                visit_counts = {}
                intrinsic_episode = 0.0
                env_reward_episode = 0.0

            steps_this_frame = RAPID_STEPS if rapid_mode else 1
            for _ in range(steps_this_frame):
                visits = visit_counts.get(s, 0) + 1
                visit_counts[s] = visits
                intrinsic = 0.0
                if current_level == 6 and intrinsic_enabled:
                    intrinsic = INTRINSIC_REWARD_STRENGTH / math.sqrt(visits + 1)
                if current_algo == "q":
                    a = epsilon_greedy(qtab, s, eps)
                    res = env.step(a)
                    learning_reward = res.reward + intrinsic + STEP_PENALTY
                    q_learning_update(qtab, s, a, learning_reward, res.next_state,
                                      ALPHA, GAMMA, res.done)
                    s = res.next_state
                else:
                    res = env.step(a)
                    ap = epsilon_greedy(qtab, res.next_state, eps)
                    learning_reward = res.reward + intrinsic + STEP_PENALTY
                    sarsa_update(qtab, s, a, learning_reward, res.next_state, ap,
                                 ALPHA, GAMMA, res.done)
                    s = res.next_state
                    a = ap

                total_reward += learning_reward
                intrinsic_episode += intrinsic
                env_reward_episode += res.reward
                steps += 1
                if res.done or steps >= MAX_STEPS:
                    break

            should_render = show_visuals or (rapid_mode and steps % RAPID_RENDER_EVERY == 0)
            if should_render:
                render(env, current_level, current_algo.upper(), ep, steps, eps,
                       total_reward, qtab, rapid_mode, comparison)

            clock.tick(FPS_VISUAL if show_visuals else FPS_FAST)

            if res.done or steps >= MAX_STEPS:
                record_episode(metrics, steps, total_reward, intrinsic_episode,
                               env_reward_episode, res.done, env.alive, res.info)
                ep += 1
                s = env.reset()
                total_reward = 0.0
                intrinsic_episode = 0.0
                env_reward_episode = 0.0
                steps = 0
                if current_algo == "sarsa":
                    a = epsilon_greedy(qtab, s, eps)
                if ep >= EPISODES:
                    comparison[current_level][current_algo] = dict(metrics)
                    save_training_curve(current_level, current_algo, metrics,
                                        intrinsic_enabled)
                    mode = "eval"
                continue

        else:
            if not eval_initialized:
                render(env, current_level, f"EVAL-{current_algo.upper()}", ep, eval_step,
                       0.0, total_reward, qtab, False, comparison)
                draw_text(screen, "P pause. 1-7 start training. I toggles Level 6 intrinsic.", 16, GAME_H - 28, COL_TEXT, small=True)
                pygame.display.flip()
                eval_initialized = True

            if not pause_eval:
                if eval_step == 0 and total_reward == 0.0:
                    s = env.reset()
                a = greedy_action(qtab, s)
                res = env.step(a)
                s = res.next_state
                total_reward += res.reward
                eval_step += 1
                if res.done or eval_step >= MAX_STEPS:
                    s = env.reset()
                    total_reward = 0.0
                    eval_step = 0

            render(env, current_level, f"EVAL-{current_algo.upper()}", ep, eval_step,
                   0.0, total_reward, qtab, False, comparison)
            draw_text(screen, "P pause. 1-7 start training. I toggles Level 6 intrinsic.", 16, GAME_H - 28, COL_TEXT, small=True)
            draw_text(
                screen,
                "Q/S comparison for the selected level is shown on the right panel.",
                16,
                GAME_H - 48,
                COL_MUTED,
                small=True,
            )
            pygame.display.flip()
            clock.tick(FPS_VISUAL)

    pygame.quit()


if __name__ == "__main__":
    main()
