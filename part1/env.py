"""GridWorld environment: transition rules, rewards, and state encoding.

Pure Python, no Pygame dependency, so it can be trained headless (see
generate_evidence.py) or driven interactively (see app.py).
"""
import random
from dataclasses import dataclass
from typing import List

from config import MONSTER_MOVE_PROBABILITY

# Actions: 0 up, 1 right, 2 down, 3 left
ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
A_UP, A_RIGHT, A_DOWN, A_LEFT = 0, 1, 2, 3
ALL_ACTIONS = [A_UP, A_RIGHT, A_DOWN, A_LEFT]


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
        dest = (p[0] + dx, p[1] + dy)
        if not self.in_bounds(dest) or self.blocked(dest):
            return p
        return dest

    def monster_step(self, monster_pos):
        if random.random() >= MONSTER_MOVE_PROBABILITY:
            return monster_pos
        dirs = ALL_ACTIONS[:]
        random.shuffle(dirs)
        for a in dirs:
            dest = self.try_move(monster_pos, a)
            if dest != monster_pos:
                return dest
        return monster_pos

    def update_monsters(self):
        if not self.monsters:
            return
        self.monsters = [self.monster_step(m) for m in self.monsters]

    def step(self, action: int) -> StepResult:
        self.step_count += 1
        reward = 0.0
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

        # End conditions: all collectible rewards obtained
        done = (
            self.apple_mask == 0
            and (len(self.keys) == 0 or self.key_mask == 0)
            and (len(self.chests) == 0 or len(self.opened_chests) == len(self.chests))
        )

        return StepResult(self.encode_state(), reward, done, info)

    def remaining_items(self):
        return {
            "apples": bin(self.apple_mask).count("1"),
            "keys": bin(self.key_mask).count("1"),
            "chests_left": max(0, len(self.chests) - len(self.opened_chests)),
        }
