# Deep RL in a Pygame Arena

A real-time Pygame action arena wrapped in a Gym-style environment, with two
different control schemes and one trained agent for each, using Stable Baselines3.

---

## 1. Files

| File | What it does |
|---|---|
| `arena_env.py` | The game + the environment (`reset`, `step`, `render`). Both control schemes live here. |
| `train.py` | Trains one agent headless (PPO or DQN), logs to TensorBoard, saves to `models/`. |
| `evaluate.py` | Loads a saved model and plays it in the visible arena. |
| `tune.py` | Short hyperparameter sweep, writes `tuning_results.csv`. |
| `play_human.py` | Play the game yourself with the keyboard (sanity check). |
| `diagnose.py` | Reports what a trained agent actually *does* (action usage, corner-hiding). |
| `models/` | Trained models (`ppo_rotation.zip`, `ppo_direct.zip`). |
| `tensorboard/` | Training logs. |

---

## 2. Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.9+.

---

## 3. How to run

**Play the game yourself (optional check)**

```bash
python play_human.py --scheme rotation    # UP thrust, LEFT/RIGHT rotate, SPACE shoot
python play_human.py --scheme direct      # ARROWS move, SPACE shoot
```

**Train the two agents (headless, no window, this is the slow part)**

```bash
python train.py --scheme rotation --algo ppo --timesteps 700000
python train.py --scheme direct   --algo ppo --timesteps 700000
```

This produces `models/ppo_rotation.zip` and `models/ppo_direct.zip`.
Roughly 20–40 minutes each on a normal laptop CPU. **Do not train for less than
about 600k steps** — at 300k the agents fight enemies but have not yet learned to
attack spawners (see section 12). DQN is also supported
(`--algo dqn`, single environment, slower to converge).

**Watch the trained agents play**

```bash
python evaluate.py --scheme rotation --episodes 3
python evaluate.py --scheme direct   --episodes 3
```

Add `--no-render` for fast numeric evaluation only, or `--model path/to/file.zip`
to load a specific checkpoint.

**GPU / speed options**

```bash
python train.py --scheme rotation --n-envs 16 --subproc    # fastest for PPO
python train.py --scheme rotation --algo dqn --device cuda # GPU is worth it for DQN
```

See section 11 for why a GPU does not speed up PPO here.

**Monitor training**

```bash
tensorboard --logdir tensorboard
```

Then open <http://localhost:6006>. The useful curves are
`rollout/ep_rew_mean` (is the agent improving?) and `rollout/ep_len_mean`
(is it surviving longer?).

**Reproduce the hyperparameter sweep**

```bash
python tune.py --scheme rotation --timesteps 60000
```

---

## 4. The arena

* Window: 960 × 680, 30 ticks per second, continuous positions (no tile grid).
* **Player ship** — cyan triangle. Acceleration + drag physics, capped speed,
  soft bounce off the walls, 100 HP, shooting on a 0.15 s cooldown.
* **Spawners** — purple squares with a health bar, 45 HP in phase 1 (+20 per phase).
  They release an enemy every
  ~2.5 s (faster in later phases) up to a global cap of 10 enemies on screen.
* **Enemies** — red circles with a health bar. They steer straight at the player
  and deal 6 damage on contact (max once every 0.8 s per enemy).
* **Bullets** — yellow dots, 10 damage, 1.4 s lifetime, radius 6 (fat enough that
  hits are discoverable during random exploration). They damage both enemies
  and spawners.
* **Phase system** — destroy every spawner in the phase and the next phase starts
  immediately with more spawners (up to 5), tougher spawner health (+20 HP),
  faster enemies (+10 px/s) and shorter spawn intervals.
* **Frame skip of 2** — each agent action is held for 2 game ticks, so an episode is
  1200 agent decisions over 2400 game ticks (~80 s).
* **Episode ends** when the player's health reaches 0 (`terminated`) or after
  1200 agent steps (`truncated`).

Rendering is completely optional and is only switched on by passing
`render_mode="human"`, so training runs are fully headless.

---

## 5. Gym-style API

```python
env = ArenaEnv(scheme="rotation")   # or "direct"
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
env.render()                        # draws the arena, evaluation only
```

The environment follows the Gymnasium convention that Stable Baselines3 expects,
so `step()` returns a 5-tuple. The classic `done` flag is simply
`done = terminated or truncated` (as used in `evaluate.py`). `info` carries the
current phase, kills, spawners destroyed and health.

---

## 6. Observation vector (24 floats, fixed size, no pixels)

| Index | Feature | Normalisation |
|---|---|---|
| 0–1 | Player x, y | / width, / height |
| 2–3 | Player velocity x, y | / max speed |
| 4–5 | Player orientation (cos, sin) | unit |
| 6 | Distance to nearest enemy | / arena diagonal |
| 7–8 | Unit direction to nearest enemy (world frame) | unit |
| 9–10 | Same direction rotated into the ship's own frame | unit |
| 11 | Is there an enemy at all | 0/1 |
| 12 | Distance to nearest spawner | / arena diagonal |
| 13–14 | Unit direction to nearest spawner (world frame) | unit |
| 15–16 | Same direction in the ship's own frame | unit |
| 17 | Is there a spawner at all | 0/1 |
| 18 | Player health | / 100 |
| 19 | Current phase | / 10 |
| 20 | Weapon ready (cooldown elapsed) | 0/1 |
| 21 | Number of enemies alive | / cap |
| 22 | Number of spawners alive | / 5 |
| 23 | Distance to the closest wall | / half height |

Two design notes:

* Directions are given both in **world coordinates** and in the **ship's own
  frame**. The ship-frame version is what the rotation agent actually needs
  ("is the target to my left or my right?"), and it makes the aiming problem the
  same shape for both schemes.
* Everything is roughly in [-1, 1], so no observation normalisation wrapper is
  needed, and the network sees inputs of comparable scale.

---

## 7. Action sets

**Control style 1 — rotation, thrust, shoot** (`Discrete(5)`)

| 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| no-op | thrust forward | rotate left | rotate right | shoot |

The ship keeps its momentum, so the agent has to plan turns in advance. It shoots
along its heading.

**Control style 2 — direct directional movement** (`Discrete(6)`)

| 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| no-op | move up | move down | move left | move right | shoot |

Movement pushes the ship along an axis (still with acceleration and drag, so it
stays semi-continuous), and the ship shoots in the direction it last moved.

Each scheme is trained separately and saved as its own model.

---

## 8. Reward function — full explanation

All the weights sit at the top of `ArenaEnv` so they can be changed in one place.

| Event | Reward | Why this value |
|---|---|---|
| Every step | **−0.02** | Time cost. Over a full 1200-step episode it sums to −24. This value is deliberately set so that **hiding is not profitable**: a perfect corner-hider scores −24, which is worse than almost any policy that actually fights. See section 12. |
| Bullet hits anything | **+0.05** | Dense feedback for aiming. An enemy takes 2 bullets and a spawner about 5, so without this the agent only hears about shooting when a target finally dies — far too sparse to learn aim from. |
| Aimed shot fired | **+0.01** | Paid when the shot leaves within ~18° of the nearest target. This rewards the *intent* to aim even when the bullet misses, which is what makes shooting discoverable in the first place. It is smaller than the hit reward, so connecting is always worth more than merely pointing. |
| Enemy destroyed | **+2.0** | The routine, repeatable achievement. Clearly positive, but ten times smaller than a spawner, so farming the enemy stream never beats removing its source. |
| Spawner destroyed | **+20.0** | The real objective. A spawner permanently removes a source of enemies, so it is worth ten enemy kills. This ratio is what makes the agent cross the arena to attack a building instead of endlessly cleaning up its output. |
| Phase cleared | **+30.0** | Paid once every spawner is down. It rewards *finishing* rather than clearing two of three spawners and drifting. |
| Damage taken | **−0.10 per HP** | One enemy touch = 6 HP = −0.6, roughly a third of an enemy kill. Scaling with damage rather than a flat per-hit penalty makes the health bar itself meaningful. Losing all 100 HP costs −10, intentionally *less* than the death penalty, so health is a resource to spend, not something to hoard. |
| Death | **−20.0** | Terminal, and the largest single term, so a suicidal rush is never optimal. It is deliberately not larger: at −30 the agent became so death-averse that avoidance dominated everything else (section 12). |
| Shaping toward the nearest spawner | **+3.0 × (d_prev − d_now) / diagonal** | See below. |

### Justification of the shaping term

Spawners have 60+ HP and are scattered around the arena, so with sparse rewards
alone an untrained agent almost never destroys one by chance — and never learns
that spawners matter at all. The shaping term pays the agent for closing the
distance to the nearest spawner and charges it for moving away.

It is written as a **difference of potentials**, `Φ(s) = −distance / diagonal`,
which is the potential-based shaping form of Ng, Harada & Russell (1999). Two
consequences matter here:

1. **It telescopes.** Over any trajectory the shaping contributions cancel out
   except for the first and last state, so the total shaping an agent can collect
   over an episode is bounded by ±3.0 — far less than a single spawner kill. It cannot be farmed: oscillating toward
   and away from a spawner nets exactly zero.
2. **It does not change the optimal policy.** Because it is a potential
   difference, the ordering of policies under the shaped reward is the same as
   under the original reward. It only makes the good policy easier to *find*.

The term is skipped on the step where a phase changes, because the "nearest
spawner" suddenly refers to a brand new set of buildings and the distance jump
would be meaningless.

---

## 9. Deep RL setup

Both final agents are **PPO** with an MLP policy — two hidden layers of 128 units
for both the policy and the value head (`net_arch=dict(pi=[128,128], vf=[128,128])`),
tanh activations, trained on 8 parallel environments.

| Hyperparameter | Value | Reason |
|---|---|---|
| `learning_rate` | 3e-4, linearly decayed to 0 | Large steps early, small steps late; without the decay the late-training reward curve was noisier. |
| `n_steps` | 1024 (× 8 envs = 8192 per update) | Long enough to capture a meaningful slice of a 1200-step episode, small enough to update often. |
| `batch_size` | 256 | Standard for a rollout buffer of 8192 (32 minibatches per epoch). |
| `n_epochs` | 10 | Default PPO reuse; more epochs started to over-fit the buffer. |
| `gamma` | 0.995 | Episodes are long (up to 1200 decisions) and the big rewards (spawner, phase) come late. At 0.99 the effective horizon (~100 steps) is too short to connect walking toward a spawner with destroying it. |
| `gae_lambda` | 0.95 | Usual bias/variance compromise. |
| `clip_range` | 0.2 | Default; stable here. |
| `ent_coef` | 0.01 | Important in this environment: with 0.0 the agents collapsed early onto a movement-only policy and stopped pressing "shoot" at all. |
| `net_arch` | [128, 128] | 24 inputs and a mix of navigation + aiming; [64, 64] under-fitted, larger nets were slower with no gain. |

DQN is provided as an alternative (`--algo dqn`) with a 200k replay buffer,
`exploration_fraction=0.3` and `target_update_interval=2000` — a slower epsilon
decay than the default, because random play here dies in about 10 seconds and the
buffer needs a longer exploratory phase to contain any spawner kills at all.

### Hyperparameter exploration

`tune.py` runs four short PPO configurations (60k steps each) and evaluates each
one over 5 episodes, writing `tuning_results.csv`:

| Config | lr | n_steps | ent_coef | gamma | net |
|---|---|---|---|---|---|
| `baseline_defaults` | 3e-4 | 2048 | 0.0 | 0.99 | [64, 64] |
| `high_entropy` | 3e-4 | 1024 | 0.01 | 0.99 | [64, 64] |
| `high_gamma_bignet` | 3e-4 | 1024 | 0.01 | 0.995 | [128, 128] |
| `fast_lr` | 1e-3 | 1024 | 0.01 | 0.995 | [128, 128] |

The settings used in `train.py` are the `high_gamma_bignet` line (entropy bonus,
long horizon, wider network). Run the script to regenerate the comparison table
and paste your own numbers here; the sweep is also logged to TensorBoard under
`tune_<scheme>_<config>` so the four learning curves can be compared directly.

---

## 10. GPU acceleration and speed

`--device` accepts `auto`, `cpu` or `cuda`, and both algorithms pass it straight
to Stable Baselines3.

**A GPU does not speed up PPO in this project.** The policy is a 24 -> 128 -> 128 -> 5
MLP, which is far too small to keep a GPU busy; the transfer overhead per
minibatch outweighs the matrix multiplications. Stable Baselines3 defaults PPO to
CPU for exactly this reason. Profiling shows roughly 80-90% of the wall-clock time
is spent inside `ArenaEnv.step()`, which is pure Python and never touches the GPU.

**A GPU does help DQN**, which performs a gradient step every 4 environment steps
and samples from a 200k replay buffer, so it is much more network-bound:

```bash
python train.py --scheme rotation --algo dqn --device cuda --timesteps 300000
```

**The real speedup for PPO is more parallel environments**, since that attacks the
actual bottleneck:

```bash
python train.py --scheme rotation --n-envs 16 --subproc --timesteps 300000
```

`--subproc` uses `SubprocVecEnv`, giving each environment its own process so all
CPU cores are used, instead of stepping the environments one after another in a
single process. Use it when `--n-envs` is 12 or more; below that the
inter-process overhead cancels out the gain. Note that `n_steps=1024` means the
rollout buffer grows with the number of environments (16 envs = 16384 samples per
update), so consider raising `batch_size` to 512 if you go much higher.

For CUDA you need a GPU-enabled torch build; the plain `pip install torch` wheel
is CPU-only on some platforms. Get the right command from
<https://pytorch.org/get-started/locally/>. If `--device cuda` is passed without
CUDA available, `train.py` prints a warning and falls back to CPU rather than
crashing.

Evaluation always loads on CPU: it feeds one observation at a time, which is
faster on CPU, and a GPU-trained model loads onto CPU without any conversion.

---

## 12. Reward debugging: how these values were reached

The first version of the reward function produced agents that **ran into a corner
and never fired a shot**, scoring about −49 on every episode. That number decomposes
exactly as: death −30, plus 100 HP lost at −0.15/HP = −15, plus ~350 steps of time
penalty ≈ −3.5.

The cause was a **local optimum in the reward function, not a training bug**:

* A perfect corner-hider scored −25 (time penalty only), while dying scored −49.
  Avoidance was therefore a real, easily reachable +24 improvement.
* The combat rewards were effectively undiscoverable. A randomly aimed bullet
  almost never connects, so the agent rarely experienced the +0.02 hit reward and
  essentially never saw a kill. It optimised the only signal it could feel.

Four changes fixed it, each measured separately (10 deterministic episodes,
`direct` scheme):

| Version | Mean reward | Enemy kills | Spawners | Phase |
|---|---|---|---|---|
| Original reward, 300k steps | −49 | 0.0 | 0.0 | 1.0 |
| + rebalanced rewards, aim bonus, frame skip (300k) | +1.6 | 16.0 | 0.0 | 1.0 |
| + stronger spawner incentive & shaping (300k) | +3.3 | 12.8 | 0.6 | 1.0 |
| + trained to 700k steps | **+101** | 17.6 | 3.4 | 2.0 |

The same final settings on the `rotation` scheme give **+181 mean reward, 6.9
spawners destroyed, phase 3.1**.

The four changes were:

1. **Frame skip of 2.** Each action is held for 2 game ticks. This halves the
   decision horizon (1200 decisions instead of 2400) and, critically, means one
   "shoot" decision reliably produces a bullet rather than being swallowed by the
   0.15 s cooldown.
2. **An aim bonus.** Rewarding a well-pointed shot even when it misses is what
   makes the whole combat branch discoverable.
3. **Raised the time cost to −0.02 and lowered the death penalty to −20.** This
   inverts the original ranking so that hiding is no longer better than fighting.
4. **Trained for 700k rather than 300k steps.** The jump from +3.3 to +101 came
   almost entirely from training length — at 300k the agent has learned to fight
   enemies but not yet to travel to spawners.

Use `diagnose.py` to check for these failure modes yourself:

```bash
python diagnose.py --scheme rotation
```

It reports the action distribution, the fraction of time spent near a wall, and
prints explicit warnings for the three classic failures (never shooting,
corner-hiding, and single-action policy collapse). A healthy `rotation` agent
looks roughly like: thrust 42%, shoot 45%, rotate 13%, near-wall 24%.

---

## 11. Notes

* Never pass `render_mode="human"` during training — rendering caps the loop at
  30 FPS and makes training roughly 100× slower.
* Checkpoints are written to `models/checkpoints/` every 50k steps, so an
  interrupted run is not lost.
* Every random draw goes through `self.np_random`, so `reset(seed=n)` reproduces
  exactly the same arena layout.
