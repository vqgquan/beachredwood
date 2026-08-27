"""
arena_env.py
Real-time Pygame arena + Gym(nasium) environment.

The scene: one player ship, spawners that keep producing enemies,
enemies that chase the player, bullets, health bars and phases.
Destroy every spawner -> next phase (harder).

Two control schemes are supported by the same class:
  scheme="rotation" -> 5 actions (noop, thrust, rotate left, rotate right, shoot)
  scheme="direct"   -> 6 actions (noop, up, down, left, right, shoot)
"""

import math
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

# ----------------------------------------------------------------------
# Arena constants (simple physics, easy to tweak)
# ----------------------------------------------------------------------
WIDTH, HEIGHT = 960, 680
DIAG = math.hypot(WIDTH, HEIGHT)        # longest possible distance, used to normalise
FPS = 30                                 # simulation ticks per second
DT = 1.0 / FPS                           # fixed time step (seconds)
MAX_STEPS = 2400                         # game ticks, ~80 seconds per episode
FRAME_SKIP = 2                           # each agent action is held for this many ticks

# Player
P_RADIUS = 14
P_ACCEL = 700.0          # px/s^2 when thrusting / moving
P_MAX_SPEED = 320.0      # px/s
P_DRAG = 1.8             # velocity damping per second
P_ROT_SPEED = 4.0        # rad/s for the rotation scheme
P_MAX_HP = 100.0
SHOOT_COOLDOWN = 0.15    # seconds between shots

# Bullets
B_SPEED = 620.0
B_RADIUS = 6            # slightly fat bullets: hits are easier to discover
B_DAMAGE = 10.0
B_LIFETIME = 1.4         # seconds before the bullet disappears

# Enemies
E_RADIUS = 12
E_BASE_SPEED = 80.0      # +10 px/s per phase
E_MAX_HP = 20.0
E_TOUCH_DAMAGE = 6.0
E_TOUCH_COOLDOWN = 0.8   # an enemy cannot hurt the player more often than this
MAX_ENEMIES = 10         # global cap so the scene stays manageable

# Spawners
S_RADIUS = 22
S_BASE_HP = 45.0         # +20 per phase (about 5 bullets in phase 1)
S_BASE_INTERVAL = 2.5    # seconds between spawns, shrinks with the phase

# Colours
C_BG = (14, 16, 24)
C_PLAYER = (90, 220, 255)
C_BULLET = (255, 240, 150)
C_ENEMY = (255, 90, 110)
C_SPAWNER = (200, 120, 255)
C_TEXT = (235, 235, 245)


# ----------------------------------------------------------------------
# Small helper objects (plain classes, no pygame sprites needed)
# ----------------------------------------------------------------------
class Bullet:
    """A player bullet flying in a straight line."""

    def __init__(self, x, y, dx, dy):
        self.x, self.y = x, y
        self.vx, self.vy = dx * B_SPEED, dy * B_SPEED
        self.life = B_LIFETIME

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt


class Enemy:
    """An enemy that walks straight at the player and hurts it on contact."""

    def __init__(self, x, y, speed):
        self.x, self.y = x, y
        self.speed = speed
        self.hp = E_MAX_HP
        self.touch_cd = 0.0

    def update(self, dt, px, py):
        dx, dy = px - self.x, py - self.y
        d = math.hypot(dx, dy) + 1e-8
        self.x += (dx / d) * self.speed * dt
        self.y += (dy / d) * self.speed * dt
        self.touch_cd = max(0.0, self.touch_cd - dt)


class Spawner:
    """A static building that periodically releases enemies."""

    def __init__(self, x, y, hp, interval):
        self.x, self.y = x, y
        self.hp = hp
        self.max_hp = hp
        self.interval = interval
        self.timer = interval * 0.5   # first enemy comes a bit earlier

    def update(self, dt):
        """Returns True when it is time to spawn an enemy."""
        self.timer -= dt
        if self.timer <= 0.0:
            self.timer = self.interval
            return True
        return False


# ----------------------------------------------------------------------
# The environment
# ----------------------------------------------------------------------
class ArenaEnv(gym.Env):
    """Gym-style API: reset(), step(action), render()."""

    metadata = {"render_modes": ["human"], "render_fps": FPS}

    # ---- reward weights (all explained in the README) ----
    R_STEP = -0.02          # time cost: hiding for a whole episode must not pay
    R_HIT = 0.05            # a bullet connected (dense feedback for aiming)
    R_AIM = 0.01            # fired while actually lined up on a target
    R_ENEMY_KILL = 2.0      # destroyed an enemy
    R_SPAWNER_KILL = 20.0   # destroyed a spawner (the real objective)
    R_PHASE = 30.0          # cleared every spawner -> next phase
    R_DAMAGE = -0.10        # per hit point lost (one enemy touch = -0.6)
    R_DEATH = -20.0         # terminal penalty
    R_SHAPE = 3.0           # potential-based shaping toward the nearest spawner

    def __init__(self, scheme="rotation", render_mode=None,
                 max_steps=MAX_STEPS, frame_skip=FRAME_SKIP):
        super().__init__()
        assert scheme in ("rotation", "direct")
        self.scheme = scheme
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        # max_steps counts agent decisions, so divide the game length by the skip
        self.max_steps = max_steps // frame_skip

        # 5 actions for the rotation scheme, 6 for the direct scheme
        self.action_space = spaces.Discrete(5 if scheme == "rotation" else 6)
        # fixed size observation: 24 floats, no pixels
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(24,), dtype=np.float32)

        # pygame surfaces are only created when render() is called
        self.screen = None
        self.clock = None
        self.font = None

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """Start a new episode and return the first observation."""
        super().reset(seed=seed)

        self.steps = 0
        self.phase = 1
        self.px, self.py = WIDTH / 2, HEIGHT / 2
        self.vx, self.vy = 0.0, 0.0
        self.angle = 0.0                 # heading in radians (rotation scheme)
        self.aim = (1.0, 0.0)            # last facing direction (direct scheme)
        self.hp = P_MAX_HP
        self.cooldown = 0.0

        self.bullets, self.enemies = [], []
        self.spawners = []
        self._build_phase()

        self.kills = 0
        self.spawners_destroyed = 0
        self.prev_spawner_dist = self._nearest(self.spawners)[0]

        return self._get_obs(), {}

    def _build_phase(self):
        """Place the spawners of the current phase away from the player."""
        n = min(1 + self.phase, 5)   # phase 1 starts with only 2 spawners
        hp = S_BASE_HP + 20.0 * (self.phase - 1)
        interval = max(0.9, S_BASE_INTERVAL - 0.25 * (self.phase - 1))
        self.spawners = []
        while len(self.spawners) < n:
            x = self.np_random.uniform(60, WIDTH - 60)
            y = self.np_random.uniform(60, HEIGHT - 60)
            if math.hypot(x - self.px, y - self.py) > 220:   # not on top of the player
                self.spawners.append(Spawner(x, y, hp, interval))

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action):
        """Apply one action, advance the simulation, return the result.

        The action is repeated for `frame_skip` game ticks. This shortens the
        decision horizon (easier credit assignment) and lets one "shoot"
        decision actually fire a bullet instead of being lost to the cooldown.
        """
        action = int(action)
        reward = self.R_STEP
        self.steps += 1

        for _ in range(self.frame_skip):
            self.cooldown = max(0.0, self.cooldown - DT)
            reward += self._apply_action(action)
            self._move_player()
            reward += self._update_bullets()
            reward += self._update_enemies()
            self._update_spawners()
            if self.hp <= 0.0:
                break

        # phase progression: every spawner of this phase is gone
        phase_changed = False
        if not self.spawners:
            self.phase += 1
            self._build_phase()
            reward += self.R_PHASE
            phase_changed = True

        # potential-based shaping toward the nearest spawner
        dist = self._nearest(self.spawners)[0]
        if not phase_changed:
            reward += self.R_SHAPE * (self.prev_spawner_dist - dist) / DIAG
        self.prev_spawner_dist = dist

        terminated = self.hp <= 0.0
        if terminated:
            reward += self.R_DEATH
        truncated = self.steps >= self.max_steps

        info = {"phase": self.phase, "kills": self.kills,
                "spawners_destroyed": self.spawners_destroyed, "hp": self.hp}

        if self.render_mode == "human":
            self.render()

        # gymnasium 5-tuple; done = terminated or truncated
        return self._get_obs(), float(reward), terminated, truncated, info

    def _apply_action(self, action):
        """Turn the discrete action into movement / shooting. Returns a reward part."""
        if self.scheme == "rotation":
            # 0 noop | 1 thrust | 2 rotate left | 3 rotate right | 4 shoot
            if action == 1:
                self.vx += math.cos(self.angle) * P_ACCEL * DT
                self.vy += math.sin(self.angle) * P_ACCEL * DT
            elif action == 2:
                self.angle -= P_ROT_SPEED * DT
            elif action == 3:
                self.angle += P_ROT_SPEED * DT
            elif action == 4:
                return self._shoot(math.cos(self.angle), math.sin(self.angle))
        else:
            # 0 noop | 1 up | 2 down | 3 left | 4 right | 5 shoot
            move = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(action)
            if move:
                self.vx += move[0] * P_ACCEL * DT
                self.vy += move[1] * P_ACCEL * DT
                self.aim = move                    # the ship faces where it moves
                self.angle = math.atan2(move[1], move[0])
            elif action == 5:
                return self._shoot(*self.aim)
        return 0.0

    def _shoot(self, dx, dy):
        """Fire a bullet. Pays a small bonus when the shot is genuinely aimed."""
        if self.cooldown > 0.0:
            return 0.0
        self.cooldown = SHOOT_COOLDOWN
        self.bullets.append(Bullet(self.px + dx * P_RADIUS, self.py + dy * P_RADIUS, dx, dy))

        # bonus if the shot points within ~18 degrees of the closest target
        best = 0.0
        for o in self.enemies + self.spawners:
            ox, oy = o.x - self.px, o.y - self.py
            d = math.hypot(ox, oy) + 1e-8
            best = max(best, (ox / d) * dx + (oy / d) * dy)
        return self.R_AIM if best > 0.95 else 0.0

    def _move_player(self):
        """Drag, speed limit and walls."""
        self.vx -= self.vx * P_DRAG * DT
        self.vy -= self.vy * P_DRAG * DT
        sp = math.hypot(self.vx, self.vy)
        if sp > P_MAX_SPEED:
            self.vx, self.vy = self.vx / sp * P_MAX_SPEED, self.vy / sp * P_MAX_SPEED
        self.px += self.vx * DT
        self.py += self.vy * DT
        # bounce softly off the borders
        if self.px < P_RADIUS or self.px > WIDTH - P_RADIUS:
            self.px = min(max(self.px, P_RADIUS), WIDTH - P_RADIUS)
            self.vx *= -0.4
        if self.py < P_RADIUS or self.py > HEIGHT - P_RADIUS:
            self.py = min(max(self.py, P_RADIUS), HEIGHT - P_RADIUS)
            self.vy *= -0.4

    def _update_bullets(self):
        """Move bullets, apply damage, remove dead objects. Returns a reward part."""
        reward = 0.0
        alive = []
        for b in self.bullets:
            b.update(DT)
            if b.life <= 0 or not (0 <= b.x <= WIDTH and 0 <= b.y <= HEIGHT):
                continue
            hit = False
            for e in self.enemies:                       # bullet vs enemy
                if math.hypot(b.x - e.x, b.y - e.y) < E_RADIUS + B_RADIUS:
                    e.hp -= B_DAMAGE
                    reward += self.R_HIT
                    hit = True
                    if e.hp <= 0:
                        reward += self.R_ENEMY_KILL
                        self.kills += 1
                    break
            if not hit:
                for s in self.spawners:                  # bullet vs spawner
                    if math.hypot(b.x - s.x, b.y - s.y) < S_RADIUS + B_RADIUS:
                        s.hp -= B_DAMAGE
                        reward += self.R_HIT
                        hit = True
                        if s.hp <= 0:
                            reward += self.R_SPAWNER_KILL
                            self.spawners_destroyed += 1
                        break
            if not hit:
                alive.append(b)
        self.bullets = alive
        self.enemies = [e for e in self.enemies if e.hp > 0]
        self.spawners = [s for s in self.spawners if s.hp > 0]
        return reward

    def _update_enemies(self):
        """Chase the player and damage it on contact. Returns a reward part."""
        reward = 0.0
        for e in self.enemies:
            e.update(DT, self.px, self.py)
            if math.hypot(e.x - self.px, e.y - self.py) < E_RADIUS + P_RADIUS and e.touch_cd <= 0:
                self.hp -= E_TOUCH_DAMAGE
                e.touch_cd = E_TOUCH_COOLDOWN
                reward += self.R_DAMAGE * E_TOUCH_DAMAGE
        self.hp = max(0.0, self.hp)
        return reward

    def _update_spawners(self):
        """Spawners release enemies until the global cap is reached."""
        speed = E_BASE_SPEED + 10.0 * (self.phase - 1)
        for s in self.spawners:
            if s.update(DT) and len(self.enemies) < MAX_ENEMIES:
                a = self.np_random.uniform(0, 2 * math.pi)
                self.enemies.append(Enemy(s.x + math.cos(a) * 30,
                                          s.y + math.sin(a) * 30, speed))

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def _nearest(self, objects):
        """Distance and unit direction to the closest object in a list."""
        best, bx, by = DIAG, 0.0, 0.0
        for o in objects:
            dx, dy = o.x - self.px, o.y - self.py
            d = math.hypot(dx, dy)
            if d < best:
                best, bx, by = d, dx, dy
        if best >= DIAG:
            return DIAG, 0.0, 0.0
        return best, bx / (best + 1e-8), by / (best + 1e-8)

    def _get_obs(self):
        """Fixed size vector of 24 normalised floats (no pixels)."""
        e_dist, e_dx, e_dy = self._nearest(self.enemies)
        s_dist, s_dx, s_dy = self._nearest(self.spawners)

        # direction to the target expressed in the ship's own frame
        ca, sa = math.cos(-self.angle), math.sin(-self.angle)
        e_rx, e_ry = e_dx * ca - e_dy * sa, e_dx * sa + e_dy * ca
        s_rx, s_ry = s_dx * ca - s_dy * sa, s_dx * sa + s_dy * ca

        wall = min(self.px, WIDTH - self.px, self.py, HEIGHT - self.py)

        obs = [
            self.px / WIDTH, self.py / HEIGHT,                  # 0-1  position
            self.vx / P_MAX_SPEED, self.vy / P_MAX_SPEED,       # 2-3  velocity
            math.cos(self.angle), math.sin(self.angle),         # 4-5  orientation
            e_dist / DIAG, e_dx, e_dy, e_rx, e_ry,              # 6-10 nearest enemy
            1.0 if self.enemies else 0.0,                       # 11   enemy exists
            s_dist / DIAG, s_dx, s_dy, s_rx, s_ry,              # 12-16 nearest spawner
            1.0 if self.spawners else 0.0,                      # 17   spawner exists
            self.hp / P_MAX_HP,                                 # 18   player health
            self.phase / 10.0,                                  # 19   current phase
            1.0 if self.cooldown <= 0 else 0.0,                 # 20   can shoot
            len(self.enemies) / MAX_ENEMIES,                    # 21   enemy count
            len(self.spawners) / 5.0,                           # 22   spawner count
            wall / (HEIGHT / 2),                                # 23   distance to wall
        ]
        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # render (evaluation only)
    # ------------------------------------------------------------------
    def render(self):
        """Draw the arena. Only call this during evaluation, never while training."""
        if self.screen is None:
            pygame.display.init()          # display + font only (no audio needed)
            pygame.font.init()
            pygame.display.set_caption(f"RL Arena - {self.scheme} scheme")
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("consolas", 18)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill(C_BG)

        # spawners: square + health bar
        for s in self.spawners:
            pygame.draw.rect(self.screen, C_SPAWNER,
                             (s.x - S_RADIUS, s.y - S_RADIUS, S_RADIUS * 2, S_RADIUS * 2), 3)
            w = int(2 * S_RADIUS * (s.hp / s.max_hp))
            pygame.draw.rect(self.screen, C_SPAWNER, (s.x - S_RADIUS, s.y - S_RADIUS - 8, w, 4))

        # enemies: circles + health bar
        for e in self.enemies:
            pygame.draw.circle(self.screen, C_ENEMY, (int(e.x), int(e.y)), E_RADIUS)
            w = int(2 * E_RADIUS * (e.hp / E_MAX_HP))
            pygame.draw.rect(self.screen, C_ENEMY, (e.x - E_RADIUS, e.y - E_RADIUS - 6, w, 3))

        # bullets
        for b in self.bullets:
            pygame.draw.circle(self.screen, C_BULLET, (int(b.x), int(b.y)), B_RADIUS)

        # player: triangle pointing at self.angle
        pts = []
        for off in (0.0, 2.5, -2.5):
            a = self.angle + off
            r = P_RADIUS + (6 if off == 0.0 else 0)
            pts.append((self.px + math.cos(a) * r, self.py + math.sin(a) * r))
        pygame.draw.polygon(self.screen, C_PLAYER, pts)

        # HUD
        pygame.draw.rect(self.screen, (70, 70, 80), (10, 10, 200, 14))
        pygame.draw.rect(self.screen, (90, 230, 140), (10, 10, int(200 * self.hp / P_MAX_HP), 14))
        txt = (f"HP {int(self.hp)}  Phase {self.phase}  Spawners {len(self.spawners)}  "
               f"Enemies {len(self.enemies)}  Kills {self.kills}  Step {self.steps}")
        self.screen.blit(self.font.render(txt, True, C_TEXT), (10, 32))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# quick manual check: random agent, no window
if __name__ == "__main__":
    env = ArenaEnv(scheme="rotation")
    obs, _ = env.reset(seed=0)
    total = 0.0
    for _ in range(300):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        if term or trunc:
            break
    print("obs shape:", obs.shape, "| reward:", round(total, 2), "| info:", info)
