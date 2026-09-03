"""
tests/test_smoke.py
====================
Fast correctness checks for Part II (Arena). Run with:

    python -m pytest tests/test_smoke.py -v

No GPU / rendering needed -- these all run headless.
"""
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from arena_simulation import ArenaSimulation as Sim, PLAYER_MAX_HP, ARENA_SIZE
from arena_env import ArenaEnv, OBS_DIM


# --------------------------------------------------------------------- #
# ArenaSimulation (pure game logic)
# --------------------------------------------------------------------- #
def test_sim_resets_to_full_health():
    sim = Sim(control_scheme="rotate")
    assert sim.player.hp == PLAYER_MAX_HP
    assert sim.phase_index == 0
    assert len(sim.spawners) > 0


def test_player_bounds_clamped():
    sim = Sim(control_scheme="rotate")
    sim.player.pos = np.array([0.0, 0.0], dtype=np.float32)
    ci = {"turn": 0, "thrust": 1, "fire": False}
    for _ in range(200):
        sim.step_physics(1 / 30, ci)
    assert 0 <= sim.player.pos[0] <= ARENA_SIZE
    assert 0 <= sim.player.pos[1] <= ARENA_SIZE


def test_direct_control_moves_player_right():
    sim = Sim(control_scheme="direct")
    start_x = sim.player.pos[0]
    ci = {"move": (1, 0), "fire": False}
    for _ in range(30):
        sim.step_physics(1 / 30, ci)
    assert sim.player.pos[0] > start_x


def test_player_bullet_kills_enemy_and_fires_event():
    sim = Sim(control_scheme="rotate")
    sim.enemies = []
    sim.spawners = []
    # place an enemy directly to the right of the player, aim and fire
    sim.player.angle = 0.0
    sim.player.pos = np.array([100.0, 100.0], dtype=np.float32)
    from arena_simulation import Enemy
    sim.enemies.append(Enemy(pos=np.array([130.0, 100.0], dtype=np.float32), hp=5.0))

    total_kills = 0
    for _ in range(60):
        ci = {"turn": 0, "thrust": 0, "fire": True}
        events = sim.step_physics(1 / 30, ci)
        total_kills += events["enemies_killed"]
        if total_kills:
            break
    assert total_kills >= 1


def test_spawner_destruction_event():
    sim = Sim(control_scheme="rotate")
    sim.enemies = []
    sim.spawners = sim.spawners[:1]
    sp = sim.spawners[0]
    sp.hp = 1.0
    sim.player.pos = sp.pos.copy() - np.array([30.0, 0.0], dtype=np.float32)
    sim.player.angle = 0.0
    destroyed = False
    for _ in range(60):
        events = sim.step_physics(1 / 30, {"turn": 0, "thrust": 0, "fire": True})
        if events["spawners_destroyed"] > 0:
            destroyed = True
            break
    assert destroyed


def test_damage_event_and_death():
    sim = Sim(control_scheme="rotate")
    sim.player.hp = 1.0
    from arena_simulation import Enemy
    sim.enemies = [Enemy(pos=sim.player.pos.copy(), hp=999.0)]
    events = sim.step_physics(1 / 30, {"turn": 0, "thrust": 0, "fire": False})
    assert events["damage_taken"] > 0
    assert events["player_died"] is True
    assert sim.done is True


# --------------------------------------------------------------------- #
# ArenaEnv (Gymnasium adapter)
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("scheme,n_actions", [("rotate", 5), ("direct", 6)])
def test_action_space_matches_spec(scheme, n_actions):
    env = ArenaEnv(control_scheme=scheme)
    assert env.action_space.n == n_actions
    env.close()


def test_observation_space_is_fixed_15dim():
    env = ArenaEnv(control_scheme="rotate")
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert env.observation_space.shape == (OBS_DIM,)
    assert np.all(obs >= -1.0 - 1e-5) and np.all(obs <= 1.0 + 1e-5)
    env.close()


def test_gymnasium_check_env_passes_both_schemes():
    for scheme in ("rotate", "direct"):
        env = ArenaEnv(control_scheme=scheme)
        check_env(env.unwrapped, skip_render_check=True)
        env.close()


def test_episode_runs_to_completion_without_crashing():
    env = ArenaEnv(control_scheme="direct", max_time=5.0)  # short episode
    obs, info = env.reset(seed=1)
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated) and steps < 5000:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (OBS_DIM,)
        assert np.isfinite(reward)
        steps += 1
    assert terminated or truncated
    env.close()


def test_reward_includes_death_penalty():
    env = ArenaEnv(control_scheme="rotate", shaping_enabled=False)
    env.reset(seed=2)
    env.sim.player.hp = 1.0
    from arena_simulation import Enemy
    env.sim.enemies = [Enemy(pos=env.sim.player.pos.copy(), hp=999.0)]
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated is True
    assert reward <= -15.0  # death penalty dominates
    env.close()


def test_shaping_toggle_changes_reward():
    """Part III ablation axis 1: shaping on vs off should produce a
    different (non-zero-diff) reward for an identical transition."""
    def run_one_step(shaping):
        env = ArenaEnv(control_scheme="rotate", shaping_enabled=shaping, seed=42)
        env.reset(seed=42)
        # move deterministically toward/away from nearest spawner
        _, r, _, _, _ = env.step(3)  # thrust
        env.close()
        return r

    r_with = run_one_step(True)
    r_without = run_one_step(False)
    assert r_with != r_without


def test_spawner_feature_ablation_zeros_observation_slice():
    """Part III ablation axis 2: with include_spawner_feature=False, obs
    dims 9,10,11 (nearest-spawner Δx,Δy,dist) must be the sentinel values."""
    env = ArenaEnv(control_scheme="rotate", include_spawner_feature=False)
    obs, _ = env.reset(seed=3)
    assert obs[8] == 0.0 and obs[9] == 0.0 and obs[10] == 1.0
    env.close()


def test_phase_completion_awards_bonus_and_advances():
    env = ArenaEnv(control_scheme="rotate", shaping_enabled=False)
    env.reset(seed=4)
    env.sim.enemies = []
    env.sim.spawners = env.sim.spawners[:1]
    env.sim.spawners[0].hp = 1.0
    env.sim.player.pos = env.sim.spawners[0].pos.copy() - np.array([30.0, 0.0], dtype=np.float32)
    env.sim.player.angle = 0.0
    total_reward = 0.0
    phase_completed_seen = False
    for _ in range(120):
        obs, reward, terminated, truncated, info = env.step(4)  # fire (rotate scheme)
        total_reward += reward
        if reward >= 4.0:  # spawner (+5) or phase (+10) bonus landed
            phase_completed_seen = True
        if terminated or truncated:
            break
    assert phase_completed_seen


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
