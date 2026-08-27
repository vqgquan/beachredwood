#!/usr/bin/env python3
# =====================================================================
# assignment3_gridworld.py
# Task 1-3: Visual GridWorld with Q-learning and SARSA in Pygame
# =====================================================================

import json
import csv
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pygame


# -----------------------------
# Configuration loader
# -----------------------------
DEFAULT_CFG = {
    "episodes": 800,
    "alpha": 0.2,
    "gamma": 0.95,
    "epsilonStart": 1.0,
    "epsilonEnd": 0.05,
    "epsilonDecayEpisodes": 700,
    "maxStepsPerEpisode": 400,
    "fpsVisual": 30,
    "fpsFast": 1000,
    "rapidStepsPerFrame": 25,
    "rapidRenderEvery": 20,
    "monsterMoveProbability": 0.4,
    "intrinsicRewardStrength": 0.2,
    "tileSize": 48,
    "panelWidth": 320,
    "seed": 42,
}


def load_config():
    cfg = DEFAULT_CFG.copy()
    path = os.path.join(os.path.dirname(__file__), "config_level0.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
        print("Loaded config_level0.json")
    return cfg


CFG = load_config()


# Unpack config
EPISODES = int(CFG["episodes"])
ALPHA = float(CFG["alpha"])
GAMMA = float(CFG["gamma"])
EPS_START = float(CFG["epsilonStart"])
EPS_END = float(CFG["epsilonEnd"])
EPS_DECAY_EP = int(CFG["epsilonDecayEpisodes"])
MAX_STEPS = int(CFG["maxStepsPerEpisode"])
FPS_VISUAL = int(CFG["fpsVisual"])
FPS_FAST = int(CFG["fpsFast"])
RAPID_STEPS = int(CFG["rapidStepsPerFrame"])
RAPID_RENDER_EVERY = int(CFG["rapidRenderEvery"])
MONSTER_MOVE_PROBABILITY = float(CFG["monsterMoveProbability"])
INTRINSIC_REWARD_STRENGTH = float(CFG["intrinsicRewardStrength"])

STEP_PENALTY = float(CFG.get("stepPenalty", 0.0))
TILE_SIZE = int(CFG["tileSize"])
PANEL_W = int(CFG["panelWidth"])
random.seed(int(CFG["seed"]))


# -----------------------------
# Pygame window
# -----------------------------
GRID_W, GRID_H = 12, 8
GAME_W, GAME_H = GRID_W * TILE_SIZE, GRID_H * TILE_SIZE
WIDTH, HEIGHT = GAME_W + PANEL_W, GAME_H

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("GridWorld - Assignment 3 Tasks 1-3")
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


# Actions: 0 up, 1 right, 2 down, 3 left
ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
A_UP, A_RIGHT, A_DOWN, A_LEFT = 0, 1, 2, 3
ALL_ACTIONS = [A_UP, A_RIGHT, A_DOWN, A_LEFT]


# -----------------------------
# Levels 0-6
# -----------------------------
def pad_level(rows: List[str]) -> List[str]:
    return [row.ljust(GRID_W)[:GRID_W] for row in rows]


LEVELS = {
    0: pad_level(
        [
            "S           ",
            "            ",
            "        A   ",
            "        A   ",
            "        A   ",
            "        A   ",
            "        A   ",
            "        A   ",
        ]
    ),
    1: pad_level(
        [
            "            ",
            "            ",
            "     FFFFFF ",
            "S          A",
            "     FFFFFF ",
            "            ",
            "            ",
            "            ",
        ]
    ),
    2: pad_level(
        [
            "S  R   A   A",
            "   R   R    ",
            "   K   R    ",
            "   R   C   A",
            "       R    ",
            "   A       R",
            "            ",
            "            ",
        ]
    ),
    3: pad_level(
        [
            "S   A   R  A",
            "R   R   R   ",
            "K       C   ",
            "R   A   R   ",
            "    R   A   ",
            "A       R   ",
            "    F   R   ",
            "        A   ",
        ]
    ),
    4: pad_level(
        [
            "S     M   A ",
            "            ",
            "    R       ",
            "            ",
            "       R    ",
            "            ",
            "            ",
            "            ",
        ]
    ),
    5: pad_level(
        [
            "S  M      A ",
            "   R        ",
            "      M     ",
            "  A         ",
            "       R    ",
            "            ",
            "            ",
            "       A    ",
        ]
    ),
    6: pad_level(
        [
            "S           ",
            "  A     R   ",
            "      K     ",
            "    R   C A ",
            "            ",
            " A          ",
            "            ",
            "            ",
        ]
    ),
}


LEVEL_LABELS = {
    0: "Level 0 - apples only",
    1: "Level 1 - apples with hazards",
    2: "Level 2 - apples, key, chest",
    3: "Level 3 - mixed collectible layout",
    4: "Level 4 - stochastic monster",
    5: "Level 5 - two stochastic monsters",
    6: "Level 6 - intrinsic reward",
}


# -----------------------------
# Helpers
# -----------------------------
def draw_text(surface, text, x, y, color=COL_TEXT, center=False, small=False):
    img = (font_small if small else font).render(text, True, color)
    rect = img.get_rect()
    if center:
        rect.center = (x, y)
    else:
        rect.topleft = (x, y)
    surface.blit(img, rect)


def linear_epsilon(ep, start, end, decay_ep):
    if decay_ep <= 0:
        return end
    t = min(ep / decay_ep, 1.0)
    return start + t * (end - start)


def random_best_action(values):
    best = max(values)
    choices = [i for i, v in enumerate(values) if v == best]
    return random.choice(choices)


# -----------------------------
# Environment
# -----------------------------
@dataclass
class StepResult:
    next_state: tuple
    reward: float
    done: bool
    info: dict


class GridWorld:
    def __init__(self, layout: List[str]):
        self.layout = layout
        self.w, self.h = len(layout[0]), len(layout)
        self.start = (0, 0)
        self.rocks = set()
        self.fires = set()
        self.keys = []
        self.chests = []
        self.apples = []
        self.monsters = []
        self.apple_index = {}
        self.key_index = {}
        self.chest_index = {}

        for y, row in enumerate(layout):
            for x, ch in enumerate(row):
                p = (x, y)
                if ch == "S":
                    self.start = p
                elif ch == "R":
                    self.rocks.add(p)
                elif ch == "F":
                    self.fires.add(p)
                elif ch == "A":
                    self.apple_index[p] = len(self.apples)
                    self.apples.append(p)
                elif ch == "K":
                    self.key_index[p] = len(self.keys)
                    self.keys.append(p)
                elif ch == "C":
                    self.chest_index[p] = len(self.chests)
                    self.chests.append(p)
                elif ch == "M":
                    self.monsters.append(p)

        self.initial_monsters = tuple(self.monsters)
        self.reset()

    def reset(self):
        self.agent = self.start
        self.alive = True
        # Monster positions are part of the stochastic environment state.
        self.monsters = list(self.initial_monsters)
        self.collected_keys = 0
        self.opened_chests = set()
        self.apple_mask = (1 << len(self.apples)) - 1
        self.key_mask = (1 << len(self.keys)) - 1
        self.chest_mask = 0
        self.step_count = 0
        return self.encode_state()

    def encode_state(self):
        return (self.agent[0], self.agent[1], self.apple_mask, self.key_mask,
                self.chest_mask, self.monster_relative_state())

    def monster_relative_state(self, clip=3):
        if not self.monsters:
            return (0, 0)
        ax, ay = self.agent
        mx, my = min(self.monsters, key=lambda m: abs(m[0] - ax) + abs(m[1] - ay))
        dx = max(-clip, min(clip, mx - ax))
        dy = max(-clip, min(clip, my - ay))
        return (dx, dy)

    def in_bounds(self, p):
        return 0 <= p[0] < self.w and 0 <= p[1] < self.h

    def blocked(self, p):
        return p in self.rocks

    def try_move(self, p, a):
        dx, dy = ACTIONS[a]
        np = (p[0] + dx, p[1] + dy)
        if not self.in_bounds(np) or self.blocked(np):
            return p
        return np

    def monster_step(self, monster_pos):
        if random.random() >= MONSTER_MOVE_PROBABILITY:
            return monster_pos
        dirs = ALL_ACTIONS[:]
        random.shuffle(dirs)
        for a in dirs:
            np = self.try_move(monster_pos, a)
            if np != monster_pos:
                return np
        return monster_pos

    def update_monsters(self):
        if not self.monsters:
            return
        new_positions = []
        for m in self.monsters:
            new_positions.append(self.monster_step(m))
        self.monsters = new_positions

    def step(self, action: int) -> StepResult:
        self.step_count += 1
        reward = 0.0
        done = False
        info = {}

        # Agent move
        self.agent = self.try_move(self.agent, action)

        # Immediate death
        if self.agent in self.fires:
            self.alive = False
            return StepResult(self.encode_state(), reward, True, {"event": "fire_death"})
        if self.agent in self.monsters:
            self.alive = False
            return StepResult(self.encode_state(), reward, True, {"event": "monster_death"})

        # Apples
        if self.agent in self.apple_index:
            idx = self.apple_index[self.agent]
            if (self.apple_mask >> idx) & 1:
                self.apple_mask &= ~(1 << idx)
                reward += 1.0
                info["apple"] = idx

        # Keys
        if self.agent in self.key_index:
            idx = self.key_index[self.agent]
            if (self.key_mask >> idx) & 1:
                self.key_mask &= ~(1 << idx)
                self.collected_keys += 1
                info["key"] = idx

        # Chest
        if self.agent in self.chest_index:
            idx = self.chest_index[self.agent]
            if idx not in self.opened_chests and self.collected_keys > 0:
                self.opened_chests.add(idx)
                self.collected_keys -= 1
                self.chest_mask |= (1 << idx)
                reward += 2.0
                info["chest"] = idx

        # Monsters move after agent action
        self.update_monsters()
        if self.agent in self.monsters:
            self.alive = False
            return StepResult(self.encode_state(), reward, True, {"event": "monster_collision"})

        # End conditions
        if self.apple_mask == 0 and (len(self.keys) == 0 or self.key_mask == 0) and (
            len(self.chests) == 0 or len(self.opened_chests) == len(self.chests)
        ):
            done = True

        return StepResult(self.encode_state(), reward, done, info)

    def remaining_items(self):
        return {
            "apples": bin(self.apple_mask).count("1"),
            "keys": bin(self.key_mask).count("1"),
            "chests_left": max(0, len(self.chests) - len(self.opened_chests)),
        }


# -----------------------------
# Q-table and learning
# -----------------------------
class QTable:
    def __init__(self):
        self.q: Dict[Tuple[tuple, int], float] = {}

    def get(self, s, a):
        return self.q.get((s, a), 0.0)

    def set(self, s, a, v):
        self.q[(s, a)] = v

    def best_value(self, s):
        return max(self.get(s, a) for a in ALL_ACTIONS)

    def best_actions(self, s):
        vals = [self.get(s, a) for a in ALL_ACTIONS]
        m = max(vals)
        return [a for a, v in zip(ALL_ACTIONS, vals) if v == m]


def epsilon_greedy(qtab: QTable, s, eps):
    if random.random() < eps:
        return random.choice(ALL_ACTIONS)
    return random.choice(qtab.best_actions(s))


def q_learning_update(qtab: QTable, s, a, r, sp, alpha, gamma, done=False):
    current = qtab.get(s, a)
    target = r if done else r + gamma * qtab.best_value(sp)
    qtab.set(s, a, current + alpha * (target - current))


def sarsa_update(qtab: QTable, s, a, r, sp, ap, alpha, gamma, done=False):
    current = qtab.get(s, a)
    target = r if done else r + gamma * qtab.get(sp, ap)
    qtab.set(s, a, current + alpha * (target - current))


# -----------------------------
# Drawing
# -----------------------------
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


def make_metrics():
    return {
        "episodes": 0,
        "successes": 0,
        "deaths": 0,
        "timeouts": 0,
        "fire_deaths": 0,
        "monster_deaths": 0,
        "steps_total": 0,
        "successful_steps": 0,
        "return_total": 0.0,
        "intrinsic_total": 0.0,
        "return_history": [],
        "env_return_history": [],
        "status_history": [],
    }


def save_training_curve(level, algorithm, metrics, intrinsic_enabled):
    folder = os.path.join(os.path.dirname(__file__), "training_curves")
    os.makedirs(folder, exist_ok=True)
    intrinsic_label = "intrinsic" if intrinsic_enabled else "extrinsic"
    path = os.path.join(folder, f"level{level}_{algorithm}_{intrinsic_label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "env_return", "total_return", "status"])
        for episode, (env_value, total_value, status) in enumerate(
            zip(metrics["env_return_history"], metrics["return_history"],
                metrics["status_history"]), start=1
        ):
            writer.writerow([episode, env_value, total_value, status])


def reset_level_state(level_idx, algo_name, qtab, current_algo, metrics):
    env = GridWorld(LEVELS[level_idx])
    metrics.clear()
    metrics.update(make_metrics())
    return env, qtab, 0, 0.0, 0, EPS_START, False, 0, False


def reset_training_run(level_idx, algo_name, qtab, metrics):
    env = GridWorld(LEVELS[level_idx])
    metrics.clear()
    metrics.update(make_metrics())
    return env, qtab, env.reset(), 0.0, 0, EPS_START, False


# -----------------------------
# Main training loop
# -----------------------------
def run_training():
    current_level = 0
    current_algo = "q"
    intrinsic_enabled = True
    env = GridWorld(LEVELS[current_level])
    qtab = QTable()
    metrics = make_metrics()
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
    eps = EPS_START
    pause_eval = False
    eval_step = 0
    eval_initialized = False
    visit_counts = {}
    s = env.reset()
    if current_algo == "sarsa":
        a = epsilon_greedy(qtab, s, eps)

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
                    qtab = QTable()
                    metrics = make_metrics()
                    s = env.reset()
                    total_reward = 0.0
                    steps = 0
                    eps = linear_epsilon(ep, EPS_START, EPS_END, EPS_DECAY_EP)
                    if current_algo == "sarsa":
                        a = epsilon_greedy(qtab, s, eps)
                elif event.key == pygame.K_1:
                    current_level = 0
                    env = GridWorld(LEVELS[current_level])
                    qtab = QTable()
                    metrics = make_metrics()
                    ep = 0
                    mode = "train"
                    eval_initialized = False
                    s = env.reset()
                    total_reward = 0.0
                    steps = 0
                    eps = linear_epsilon(ep, EPS_START, EPS_END, EPS_DECAY_EP)
                    if current_algo == "sarsa":
                        a = epsilon_greedy(qtab, s, eps)
                elif event.key == pygame.K_2:
                    current_level = 1
                    env = GridWorld(LEVELS[current_level])
                    qtab = QTable()
                    metrics = make_metrics()
                    ep = 0
                    mode = "train"
                    eval_initialized = False
                    s = env.reset()
                    total_reward = 0.0
                    steps = 0
                    eps = linear_epsilon(ep, EPS_START, EPS_END, EPS_DECAY_EP)
                    if current_algo == "sarsa":
                        a = epsilon_greedy(qtab, s, eps)
                elif event.key == pygame.K_3:
                    current_level = 2
                    env = GridWorld(LEVELS[current_level])
                    qtab = QTable()
                    metrics = make_metrics()
                    ep = 0
                    mode = "train"
                    eval_initialized = False
                    s = env.reset()
                    total_reward = 0.0
                    steps = 0
                    eps = linear_epsilon(ep, EPS_START, EPS_END, EPS_DECAY_EP)
                    if current_algo == "sarsa":
                        a = epsilon_greedy(qtab, s, eps)
                elif event.key == pygame.K_4:
                    current_level = 3
                    env = GridWorld(LEVELS[current_level])
                    qtab = QTable()
                    metrics = make_metrics()
                    ep = 0
                    mode = "train"
                    eval_initialized = False
                    s = env.reset()
                    total_reward = 0.0
                    steps = 0
                    eps = linear_epsilon(ep, EPS_START, EPS_END, EPS_DECAY_EP)
                    if current_algo == "sarsa":
                        a = epsilon_greedy(qtab, s, eps)
                elif pygame.K_5 <= event.key <= pygame.K_7:
                    current_level = event.key - pygame.K_1
                    env = GridWorld(LEVELS[current_level])
                    qtab = QTable()
                    metrics = make_metrics()
                    ep = 0
                    mode = "train"
                    eval_initialized = False
                    intrinsic_enabled = current_level == 6
                    s = env.reset()
                    total_reward = 0.0
                    intrinsic_episode = 0.0
                    env_reward_episode = 0.0
                    steps = 0
                    eps = linear_epsilon(ep, EPS_START, EPS_END, EPS_DECAY_EP)
                    if current_algo == "sarsa":
                        a = epsilon_greedy(qtab, s, eps)
                elif event.key == pygame.K_i and current_level == 6:
                    intrinsic_enabled = not intrinsic_enabled
                    ep = 0
                    qtab = QTable()
                    metrics = make_metrics()
                    env = GridWorld(LEVELS[current_level])
                    s = env.reset()
                    total_reward = 0.0
                    intrinsic_episode = 0.0
                    env_reward_episode = 0.0
                    steps = 0
                    eps = EPS_START
                    mode = "train"
                    eval_initialized = False
                    if current_algo == "sarsa":
                        a = epsilon_greedy(qtab, s, eps)
                elif event.key == pygame.K_q:
                    current_algo = "q"
                    ep = 0
                    qtab = QTable()
                    metrics = make_metrics()
                    env = GridWorld(LEVELS[current_level])
                    s = env.reset()
                    total_reward = 0.0
                    steps = 0
                    eps = EPS_START
                    mode = "train"
                    eval_initialized = False
                elif event.key == pygame.K_s:
                    current_algo = "sarsa"
                    ep = 0
                    qtab = QTable()
                    metrics = make_metrics()
                    env = GridWorld(LEVELS[current_level])
                    s = env.reset()
                    total_reward = 0.0
                    steps = 0
                    eps = EPS_START
                    mode = "train"
                    eval_initialized = False
                    a = epsilon_greedy(qtab, s, eps)

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
            for event in pygame.event.get():
                pass

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
                metrics["episodes"] += 1
                metrics["steps_total"] += steps
                metrics["return_total"] += total_reward
                metrics["intrinsic_total"] += intrinsic_episode
                metrics["return_history"].append(total_reward)
                metrics["env_return_history"].append(env_reward_episode)
                if res.done and env.alive:
                    metrics["status_history"].append("success")
                elif res.done:
                    metrics["status_history"].append("death")
                else:
                    metrics["status_history"].append("timeout")
                if res.done and env.alive:
                    metrics["successes"] += 1
                    metrics["successful_steps"] += steps
                elif res.done:
                    metrics["deaths"] += 1
                    if res.info.get("event") == "fire_death":
                        metrics["fire_deaths"] += 1
                    elif res.info.get("event") in ("monster_death", "monster_collision"):
                        metrics["monster_deaths"] += 1
                else:
                    metrics["timeouts"] += 1
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
                a = random.choice(qtab.best_actions(s))
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
    run_training()
