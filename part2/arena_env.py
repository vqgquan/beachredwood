"""
arena_env.py
============
Thin Gymnasium adapter around ArenaSimulation. This is the ONLY module that
knows about RL concepts (observation vectors, action spaces, reward
shaping, episode termination semantics). Game rules live in
arena_simulation.py.

Reward (matches architecture doc, potential-based shaping per
Ng, Harada & Russell 1999 -- provably preserves the optimal policy):

    +1.0   bullet hits an enemy
    +5.0   spawner destroyed
    +10.0  phase completed
    -0.5   player takes damage (per damage EVENT, not per hp point)
    -15.0  player dies (terminal)
    + shaping: r_shape = gamma * Phi(s') - Phi(s)
               Phi(s) = -k * dist(player, nearest_active_spawner)

`k` and `gamma` are constructor args so Part III's ablation study (shaping
on/off) can toggle this cleanly by setting shaping_enabled=False.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from arena_simulation import (
    ArenaSimulation, ARENA_SIZE, MAX_SPEED, PLAYER_MAX_HP,
    MAX_EPISODE_TIME, MAX_ENEMIES_EXPECTED, DEFAULT_PHASES,
)

OBS_DIM = 15

# Reward constants -- kept as module-level defaults, overridable per-instance
R_ENEMY_HIT = 1.0
R_SPAWNER_DESTROYED = 5.0
R_PHASE_COMPLETE = 10.0
R_DAMAGE_TAKEN = -0.5
R_DEATH = -15.0


class ArenaEnv(gym.Env):
    """Gymnasium environment for the Arena mini-game.

    Parameters
    ----------
    control_scheme : "rotate" | "direct"
        "rotate"  -> Discrete(5): noop, turn_left, turn_right, thrust, fire
        "direct"  -> Discrete(6): noop, up, down, left, right, fire
    shaping_enabled : bool
        Toggle potential-based reward shaping. Used by the Part III
        ablation study (with/without shaping).
    include_spawner_feature : bool
        If False, the 3 "nearest spawner" observation dims (Δx,Δy,dist)
        are zeroed out. Used by the Part III ablation study
        (full observation vs. spawner-feature ablated).
    shaping_k : float
        Potential function scale k in Phi(s) = -k * dist(...).
    max_time : float
        Episode time budget in seconds (drives truncation).
    render_mode : "human" | None
        Only set to "human" in eval scripts; training must stay headless.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 control_scheme: str = "rotate",
                 shaping_enabled: bool = True,
                 include_spawner_feature: bool = True,
                 shaping_k: float = 0.01,
                 gamma: float = 0.99,
                 max_time: float = MAX_EPISODE_TIME,
                 phases: Optional[list] = None,
                 render_mode: Optional[str] = None,
                 seed: Optional[int] = None):
        super().__init__()
        assert control_scheme in ("rotate", "direct")
        self.control_scheme = control_scheme
        self.shaping_enabled = shaping_enabled
        self.include_spawner_feature = include_spawner_feature
        self.shaping_k = shaping_k
        self.gamma = gamma
        self.render_mode = render_mode

        self.sim = ArenaSimulation(
            control_scheme=control_scheme,
            phases=phases if phases is not None else DEFAULT_PHASES,
            max_time=max_time,
        )
        if seed is not None:
            self.sim.rng.seed(seed)

        self.action_space = spaces.Discrete(5 if control_scheme == "rotate" else 6)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self._dt = 1.0 / 30.0  # fixed physics timestep (30 Hz)
        self._prev_potential = 0.0
        self._screen = None  # lazily created pygame surface for render()

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.sim.rng.seed(seed)
        self.sim.reset()
        obs = self._build_observation()
        self._prev_potential = self._potential()
        return obs, {}

    def step(self, action: int):
        control_input = self._action_to_control(action)
        events = self.sim.step_physics(self._dt, control_input)

        reward = 0.0
        reward += events["enemies_killed"] * R_ENEMY_HIT
        reward += events["spawners_destroyed"] * R_SPAWNER_DESTROYED
        if events["phase_completed"]:
            reward += R_PHASE_COMPLETE
        if events["damage_taken"] > 0:
            reward += R_DAMAGE_TAKEN
        if events["player_died"]:
            reward += R_DEATH

        if self.shaping_enabled:
            new_potential = self._potential()
            reward += self.gamma * new_potential - self._prev_potential
            self._prev_potential = new_potential

        terminated = bool(events["player_died"] or events["episode_won"])
        truncated = bool(self.sim.done and not terminated)  # timeout path

        obs = self._build_observation()
        info = {
            "phase_index": self.sim.phase_index,
            "episode_won": events["episode_won"],
            "enemies_killed": events["enemies_killed"],
            "spawners_destroyed": events["spawners_destroyed"],
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _action_to_control(self, action: int) -> dict:
        if self.control_scheme == "rotate":
            # 0 noop, 1 turn_left, 2 turn_right, 3 thrust, 4 fire
            return {
                "turn": -1 if action == 1 else (1 if action == 2 else 0),
                "thrust": 1 if action == 3 else 0,
                "fire": action == 4,
            }
        else:
            # 0 noop, 1 up, 2 down, 3 left, 4 right, 5 fire
            move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            return {
                "move": move_map.get(action, (0, 0)),
                "fire": action == 5,
            }

    # ------------------------------------------------------------------ #
    def _nearest(self, points: list, origin: np.ndarray):
        """Return (dx, dy, dist) to nearest point in `points`, or a
        sentinel "far away" value if the list is empty."""
        if not points:
            # sentinel: maximum possible distance, zero direction
            return 0.0, 0.0, 1.0
        dists = [float(np.linalg.norm(pt - origin)) for pt in points]
        idx = int(np.argmin(dists))
        nearest = points[idx]
        d = dists[idx]
        max_d = ARENA_SIZE * math.sqrt(2)
        dx = _norm(nearest[0] - origin[0], ARENA_SIZE)
        dy = _norm(nearest[1] - origin[1], ARENA_SIZE)
        return dx, dy, _clamp01(d / max_d)

    def _build_observation(self) -> np.ndarray:
        s = self.sim.get_state_dict()
        px, py = s["player_pos"]
        vx, vy = s["player_vel"]

        edx, edy, edist = self._nearest(s["enemies"], s["player_pos"])
        if self.include_spawner_feature:
            sdx, sdy, sdist = self._nearest(s["spawners"], s["player_pos"])
        else:
            sdx, sdy, sdist = 0.0, 0.0, 1.0

        obs = np.array([
            _norm(px, ARENA_SIZE),                                   # 1 player x
            _norm(py, ARENA_SIZE),                                   # 2 player y
            _clamp(vx / MAX_SPEED, -1.0, 1.0),                       # 3 vx
            _clamp(vy / MAX_SPEED, -1.0, 1.0),                       # 4 vy
            _clamp(s["player_angle"] / math.pi, -1.0, 1.0),          # 5 angle
            edx, edy, edist,                                        # 6-8 nearest enemy
            sdx, sdy, sdist,                                         # 9-11 nearest spawner
            _clamp01(s["player_hp"] / PLAYER_MAX_HP) * 2 - 1,        # 12 health
            _clamp01(s["n_enemies"] / MAX_ENEMIES_EXPECTED) * 2 - 1,  # 13 enemy count
            _clamp01(s["phase_index"] / max(1, s["n_phases"] - 1)) * 2 - 1,  # 14 phase
            _clamp01(s["time_remaining"] / self.sim.max_time) * 2 - 1,  # 15 time left
        ], dtype=np.float32)
        return obs

    def _potential(self) -> float:
        """Phi(s) = -k * dist(player, nearest active spawner).
        0 if no spawners remain (about to transition phase / episode end)
        or if the spawner feature is ablated (Part III option)."""
        if not self.include_spawner_feature:
            return 0.0
        s = self.sim.get_state_dict()
        if not s["spawners"]:
            return 0.0
        dists = [float(np.linalg.norm(sp - s["player_pos"])) for sp in s["spawners"]]
        return -self.shaping_k * min(dists)

    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode != "human":
            return
        import pygame
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((int(ARENA_SIZE), int(ARENA_SIZE)))
            pygame.display.set_caption(f"Arena RL — {self.control_scheme}")
        self.sim.render(self._screen)
        pygame.display.flip()

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None


def _norm(v: float, scale: float) -> float:
    """Map a coordinate/delta in [-scale, scale] to [-1, 1]."""
    return float(_clamp(v / scale, -1.0, 1.0))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _clamp01(v) -> float:
    return max(0.0, min(1.0, v))
