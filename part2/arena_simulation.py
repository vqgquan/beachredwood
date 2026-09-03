"""
arena_simulation.py
====================
Pure game-logic layer for the Assignment 3 / Part II "Arena" mini-game.

Design intent (per architecture doc): this module knows NOTHING about
reinforcement learning. No observation vectors, no reward shaping, no
Gymnasium imports. It only implements:

    Player, Enemy, Spawner, Bullet   -- entities
    ArenaSimulation                  -- physics/rules loop + Pygame rendering

`ArenaEnv` (in arena_env.py) is the thin RL adapter that drives this class
via `step_physics()` and reads `get_state_dict()` to build observations.

Coordinate system: arena is a fixed-size square, origin top-left, x right,
y down (matches Pygame convention). ARENA_SIZE is in "world units" so the
same numbers can be reused for observation normalisation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Constants (also imported by arena_env.py for observation normalisation)
# --------------------------------------------------------------------------- #
ARENA_SIZE = 600.0          # world is ARENA_SIZE x ARENA_SIZE
MAX_SPEED = 220.0           # world units / sec, used to normalise vx,vy
PLAYER_RADIUS = 12.0
ENEMY_RADIUS = 10.0
SPAWNER_RADIUS = 18.0
BULLET_RADIUS = 3.0
BULLET_SPEED = 380.0
ENEMY_BULLET_SPEED = 240.0

PLAYER_MAX_HP = 100.0
ENEMY_MAX_HP = 20.0
SPAWNER_MAX_HP = 60.0

PLAYER_ACCEL = 420.0         # world units / sec^2 while thrusting
PLAYER_DRAG = 1.8            # exponential velocity damping per second
TURN_RATE = math.pi * 1.4    # rad/sec for "rotate" control scheme
DIRECT_MOVE_SPEED = 200.0    # target speed for "direct" control scheme

FIRE_COOLDOWN = 0.28         # sec between player shots
ENEMY_FIRE_COOLDOWN = 1.6    # sec between an enemy's shots
ENEMY_SPEED = 70.0
ENEMY_CONTACT_DAMAGE = 8.0
ENEMY_CONTACT_COOLDOWN = 0.6
ENEMY_BULLET_DAMAGE = 6.0

MAX_EPISODE_TIME = 60.0      # sec, used to normalise "time remaining"
MAX_ENEMIES_EXPECTED = 12.0  # used to normalise "enemy count" in obs


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #
@dataclass
class Bullet:
    pos: np.ndarray
    vel: np.ndarray
    owner: str  # "player" | "enemy"
    radius: float = BULLET_RADIUS
    damage: float = 10.0
    alive: bool = True


@dataclass
class Enemy:
    pos: np.ndarray
    hp: float = ENEMY_MAX_HP
    radius: float = ENEMY_RADIUS
    speed: float = ENEMY_SPEED
    fire_timer: float = field(default_factory=lambda: random.uniform(0.0, ENEMY_FIRE_COOLDOWN))
    contact_cd: float = 0.0
    alive: bool = True
    phase_index: int = 0  # which phase spawned this enemy (for clear-condition bookkeeping)


@dataclass
class Spawner:
    pos: np.ndarray
    hp: float = SPAWNER_MAX_HP
    radius: float = SPAWNER_RADIUS
    spawn_interval: float = 3.0
    spawn_timer: float = 0.0
    max_concurrent: int = 3
    phase_index: int = 0
    alive: bool = True


@dataclass
class Player:
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    angle: float = -math.pi / 2  # facing "up" initially
    hp: float = PLAYER_MAX_HP
    radius: float = PLAYER_RADIUS
    fire_cd: float = 0.0


@dataclass
class PhaseDef:
    """Static definition of one arena phase: where spawners appear and how
    tough enemies from this phase are. Phase clears when every spawner
    belonging to it is destroyed AND every enemy it produced is dead."""
    spawner_positions: list
    spawn_interval: float = 3.0
    max_concurrent_per_spawner: int = 3
    enemy_hp: float = ENEMY_MAX_HP
    enemy_speed: float = ENEMY_SPEED


DEFAULT_PHASES = [
    PhaseDef(spawner_positions=[(150, 150)], spawn_interval=3.0,
             max_concurrent_per_spawner=2, enemy_hp=16.0, enemy_speed=60.0),
    PhaseDef(spawner_positions=[(450, 150), (150, 450)], spawn_interval=2.6,
             max_concurrent_per_spawner=2, enemy_hp=20.0, enemy_speed=72.0),
    PhaseDef(spawner_positions=[(450, 450), (300, 100), (100, 300)], spawn_interval=2.2,
             max_concurrent_per_spawner=3, enemy_hp=24.0, enemy_speed=85.0),
]


class ArenaSimulation:
    """Pure game-logic simulation. No RL, no gym. Optionally renders via Pygame.

    control_scheme: "rotate" (thrust + turn, Asteroids-style) or
                     "direct" (4-directional velocity control).
    This only affects how `control_input` dicts passed to `step_physics`
    are interpreted -- it is a physics/gameplay concept, not an RL one.
    """

    def __init__(self, control_scheme: str = "rotate",
                 phases: Optional[list] = None,
                 max_time: float = MAX_EPISODE_TIME,
                 rng: Optional[random.Random] = None):
        assert control_scheme in ("rotate", "direct")
        self.control_scheme = control_scheme
        self.phases = phases if phases is not None else DEFAULT_PHASES
        self.max_time = max_time
        self.rng = rng or random.Random()

        self.player: Player = None
        self.enemies: list[Enemy] = []
        self.spawners: list[Spawner] = []
        self.bullets: list[Bullet] = []
        self.phase_index: int = 0
        self.time_elapsed: float = 0.0
        self.done: bool = False
        self.win: bool = False

        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.player = Player(pos=np.array([ARENA_SIZE / 2, ARENA_SIZE / 2], dtype=np.float32))
        self.enemies = []
        self.spawners = []
        self.bullets = []
        self.phase_index = 0
        self.time_elapsed = 0.0
        self.done = False
        self.win = False
        self._spawn_phase(0)

    def _spawn_phase(self, idx: int) -> None:
        pdef = self.phases[idx]
        for (x, y) in pdef.spawner_positions:
            self.spawners.append(Spawner(
                pos=np.array([x, y], dtype=np.float32),
                spawn_interval=pdef.spawn_interval,
                max_concurrent=pdef.max_concurrent_per_spawner,
                phase_index=idx,
                spawn_timer=self.rng.uniform(0.0, pdef.spawn_interval),
            ))

    # ------------------------------------------------------------------ #
    def step_physics(self, dt: float, control_input: dict) -> dict:
        """Advance the simulation by dt seconds given a control_input dict.

        control_input keys depend on control_scheme:
          "rotate": {"turn": -1|0|1, "thrust": 0|1, "fire": bool}
          "direct": {"move": (dx, dy) each in {-1,0,1}, "fire": bool}

        Returns a dict of raw game EVENTS this tick (no reward numbers --
        ArenaEnv turns these into rewards):
          enemies_killed: int
          spawners_destroyed: int
          phase_completed: bool
          damage_taken: float          (0 if none)
          player_died: bool
          episode_won: bool
        """
        events = {
            "enemies_killed": 0,
            "spawners_destroyed": 0,
            "phase_completed": False,
            "damage_taken": 0.0,
            "player_died": False,
            "episode_won": False,
        }
        if self.done:
            return events

        self.time_elapsed += dt
        self._apply_player_control(dt, control_input)
        self._update_bullets(dt)
        self._update_enemies(dt, events)
        self._update_spawners(dt)
        self._handle_collisions(events)
        self._check_phase_clear(events)

        if self.player.hp <= 0 and not self.done:
            self.player.hp = 0
            self.done = True
            self.win = False
            events["player_died"] = True

        if self.time_elapsed >= self.max_time and not self.done:
            self.done = True  # truncation, handled by ArenaEnv as time-out

        return events

    # ------------------------------------------------------------------ #
    def _apply_player_control(self, dt: float, ci: dict) -> None:
        p = self.player
        p.fire_cd = max(0.0, p.fire_cd - dt)

        if self.control_scheme == "rotate":
            turn = _clamp(ci.get("turn", 0), -1, 1)
            thrust = 1.0 if ci.get("thrust", 0) else 0.0
            p.angle += turn * TURN_RATE * dt
            if thrust:
                p.vel[0] += math.cos(p.angle) * PLAYER_ACCEL * dt
                p.vel[1] += math.sin(p.angle) * PLAYER_ACCEL * dt
        else:  # "direct"
            mx, my = ci.get("move", (0, 0))
            direction = np.array([mx, my], dtype=np.float32)
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
                p.vel[0] += direction[0] * PLAYER_ACCEL * dt
                p.vel[1] += direction[1] * PLAYER_ACCEL * dt
                p.angle = math.atan2(direction[1], direction[0])

        # drag + speed clamp
        drag_factor = math.exp(-PLAYER_DRAG * dt)
        p.vel *= drag_factor
        speed = float(np.linalg.norm(p.vel))
        if speed > MAX_SPEED:
            p.vel = p.vel / speed * MAX_SPEED

        p.pos += p.vel * dt
        p.pos[0] = _clamp(p.pos[0], p.radius, ARENA_SIZE - p.radius)
        p.pos[1] = _clamp(p.pos[1], p.radius, ARENA_SIZE - p.radius)

        if ci.get("fire", False) and p.fire_cd <= 0.0:
            direction = np.array([math.cos(p.angle), math.sin(p.angle)], dtype=np.float32)
            self.bullets.append(Bullet(
                pos=p.pos.copy() + direction * (p.radius + BULLET_RADIUS),
                vel=direction * BULLET_SPEED,
                owner="player",
            ))
            p.fire_cd = FIRE_COOLDOWN

    # ------------------------------------------------------------------ #
    def _update_bullets(self, dt: float) -> None:
        for b in self.bullets:
            b.pos += b.vel * dt
            if not (0 <= b.pos[0] <= ARENA_SIZE and 0 <= b.pos[1] <= ARENA_SIZE):
                b.alive = False
        self.bullets = [b for b in self.bullets if b.alive]

    def _update_enemies(self, dt: float, events: dict) -> None:
        p = self.player
        for e in self.enemies:
            if not e.alive:
                continue
            to_player = p.pos - e.pos
            dist = float(np.linalg.norm(to_player))
            if dist > 1e-6:
                e.pos += (to_player / dist) * e.speed * dt

            e.fire_timer -= dt
            if e.fire_timer <= 0.0 and dist > 1e-6:
                direction = to_player / dist
                self.bullets.append(Bullet(
                    pos=e.pos.copy() + direction * (e.radius + BULLET_RADIUS),
                    vel=direction * ENEMY_BULLET_SPEED,
                    owner="enemy",
                    damage=ENEMY_BULLET_DAMAGE,
                ))
                e.fire_timer = ENEMY_FIRE_COOLDOWN

            e.contact_cd = max(0.0, e.contact_cd - dt)
            if dist <= (e.radius + p.radius) and e.contact_cd <= 0.0:
                p.hp -= ENEMY_CONTACT_DAMAGE
                events["damage_taken"] += ENEMY_CONTACT_DAMAGE
                e.contact_cd = ENEMY_CONTACT_COOLDOWN

        self.enemies = [e for e in self.enemies if e.alive]

    def _update_spawners(self, dt: float) -> None:
        for s in self.spawners:
            if not s.alive:
                continue
            n_alive_from_this = sum(
                1 for e in self.enemies if e.alive and e.phase_index == s.phase_index
            )
            s.spawn_timer -= dt
            if s.spawn_timer <= 0.0 and n_alive_from_this < s.max_concurrent:
                pdef = self.phases[s.phase_index]
                self.enemies.append(Enemy(
                    pos=s.pos.copy(),
                    hp=pdef.enemy_hp,
                    speed=pdef.enemy_speed,
                    phase_index=s.phase_index,
                ))
                s.spawn_timer = s.spawn_interval

    def _handle_collisions(self, events: dict) -> None:
        p = self.player
        for b in self.bullets:
            if not b.alive:
                continue
            if b.owner == "player":
                for e in self.enemies:
                    if e.alive and _dist(b.pos, e.pos) <= (b.radius + e.radius):
                        e.hp -= b.damage
                        b.alive = False
                        if e.hp <= 0:
                            e.alive = False
                            events["enemies_killed"] += 1
                        break
                if b.alive:
                    for s in self.spawners:
                        if s.alive and _dist(b.pos, s.pos) <= (b.radius + s.radius):
                            s.hp -= b.damage
                            b.alive = False
                            if s.hp <= 0:
                                s.alive = False
                                events["spawners_destroyed"] += 1
                            break
            elif b.owner == "enemy":
                if _dist(b.pos, p.pos) <= (b.radius + p.radius):
                    p.hp -= b.damage
                    events["damage_taken"] += b.damage
                    b.alive = False

        self.bullets = [b for b in self.bullets if b.alive]
        self.enemies = [e for e in self.enemies if e.alive]

    def _check_phase_clear(self, events: dict) -> None:
        if self.done:
            return
        current_spawners = [s for s in self.spawners if s.phase_index == self.phase_index]
        current_enemies = [e for e in self.enemies if e.phase_index == self.phase_index]
        spawners_clear = all(not s.alive for s in current_spawners) if current_spawners else False
        enemies_clear = not any(e.alive for e in current_enemies)
        if spawners_clear and enemies_clear:
            events["phase_completed"] = True
            if self.phase_index + 1 < len(self.phases):
                self.phase_index += 1
                self._spawn_phase(self.phase_index)
            else:
                self.done = True
                self.win = True
                events["episode_won"] = True

    # ------------------------------------------------------------------ #
    def get_state_dict(self) -> dict:
        """Raw numeric state for the observation builder in arena_env.py.
        Intentionally NOT a gym observation -- just plain game state."""
        return {
            "player_pos": self.player.pos.copy(),
            "player_vel": self.player.vel.copy(),
            "player_angle": self.player.angle,
            "player_hp": self.player.hp,
            "enemies": [(e.pos.copy()) for e in self.enemies if e.alive],
            "spawners": [(s.pos.copy()) for s in self.spawners if s.alive],
            "n_enemies": sum(1 for e in self.enemies if e.alive),
            "phase_index": self.phase_index,
            "n_phases": len(self.phases),
            "time_remaining": max(0.0, self.max_time - self.time_elapsed),
        }

    # ------------------------------------------------------------------ #
    def render(self, surface) -> None:
        """Draw current state onto a Pygame surface. Only called during
        eval scripts (render=True) -- never during headless training."""
        import pygame

        surface.fill((15, 15, 25))

        for s in self.spawners:
            if s.alive:
                frac = _clamp(s.hp / SPAWNER_MAX_HP, 0.0, 1.0)
                color = (int(200 * (1 - frac) + 40 * frac), int(80 + 100 * frac), 220)
                pygame.draw.circle(surface, color, s.pos.astype(int), int(s.radius))
                pygame.draw.circle(surface, (255, 255, 255), s.pos.astype(int), int(s.radius), 2)

        for e in self.enemies:
            if e.alive:
                pygame.draw.circle(surface, (230, 70, 70), e.pos.astype(int), int(e.radius))

        for b in self.bullets:
            color = (250, 230, 90) if b.owner == "player" else (255, 120, 120)
            pygame.draw.circle(surface, color, b.pos.astype(int), int(b.radius))

        p = self.player
        tip = p.pos + np.array([math.cos(p.angle), math.sin(p.angle)]) * p.radius
        pygame.draw.circle(surface, (90, 200, 255), p.pos.astype(int), int(p.radius))
        pygame.draw.line(surface, (255, 255, 255), p.pos.astype(int), tip.astype(int), 2)

        # HUD
        font = pygame.font.SysFont("consolas", 16)
        hud = (f"HP {int(p.hp)}/{int(PLAYER_MAX_HP)}  "
               f"Phase {self.phase_index + 1}/{len(self.phases)}  "
               f"t={self.time_elapsed:4.1f}s")
        surface.blit(font.render(hud, True, (255, 255, 255)), (8, 8))
