# Part I — Classical RL in a Pygame Gridworld

Tabular Q-learning and SARSA agents learning to collect apples, use keys on
chests, and avoid instant-death hazards (fire, monsters) in a 12x8 Pygame
gridworld, across 7 levels of increasing difficulty.

---

## 1. Files

| File                   | What it does                                                                                                                     |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`            | Loads `config.json` (hyperparameters), seeds the RNG.                                                                            |
| `levels.py`            | The 7 level layouts (ASCII grids) and their labels.                                                                              |
| `env.py`               | `GridWorld`: transition rules, rewards, death conditions, state encoding. No Pygame dependency.                                  |
| `agents.py`            | `QTable`, epsilon-greedy action selection (with random tie-breaking), the Q-learning and SARSA update rules.                     |
| `metrics.py`           | Per-run bookkeeping (`make_metrics`, `record_episode`) and CSV export, shared by the interactive app and the headless generator. |
| `app.py`               | **The graded deliverable.** Interactive Pygame window: trains live, on screen, with keyboard control over level/algorithm/speed. |
| `generate_evidence.py` | Headless (no window) re-run of all 16 level x algorithm x intrinsic-setting combinations, for reproducible report evidence.      |
| `tune_intrinsic.py`    | Sweeps `intrinsicRewardStrength` on Level 6 to justify the Task 5 write-up.                                                      |
| `plot_curves.py`       | Turns the CSVs/JSON above into the PNG charts under `training_curves/plots/`.                                                    |
| `config.json`          | Hyperparameters (episodes, alpha, gamma, epsilon schedule, etc.) — shared by all levels.                                         |
| `training_curves/`     | CSV learning curves, `summary.csv`, greedy-rollout paths, and rendered PNGs (report evidence).                                   |

---

## 2. Installation

Requires **Python 3.13** — pygame does not yet ship a prebuilt wheel for 3.13+
newer interpreters (e.g. 3.14) on macOS, so installing it there tries to
compile from source and fails on a missing `SDL.h`.

```bash
python3.13 -m venv .venv          # from the repo root, shared with part2
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r part1/requirements.txt
```

---

## 3. How to run

**Play/train interactively (the graded deliverable)**

```bash
cd part1
python app.py
```

Controls: `1`-`7` pick a level, `Q`/`S` pick the algorithm, `I` toggles the
Level 6 intrinsic reward, `T` toggles rapid (mostly-unrendered) training
speed, `V` toggles per-step rendering, `P` pauses evaluation playback, `R`
soft-resets the Q-table without resetting the episode/epsilon schedule, `Esc`
quits. Training runs live on screen; once `episodes` (from `config.json`) is
reached it switches to a looping greedy-policy evaluation playback and writes
`training_curves/level{N}_{algo}_{intrinsic|extrinsic}.csv`.

**Regenerate all report evidence headlessly** (no window, ~10 seconds)

Should only run when changes in

- env.py, agents.py (rules or update logic)
- config.json (episodes, alpha, gamma, epsilon schedule, monster probability, intrinsic strength, seed, etc.)
- level layout in levels.py

```bash
cd part1
python generate_evidence.py   # trains all 16 runs, writes CSVs + summary.csv + greedy paths
python tune_intrinsic.py      # intrinsic-reward-strength sweep for Level 6
python plot_curves.py         # renders everything above to PNG under training_curves/plots/
```

These three scripts import the exact same `env.py`/`agents.py` functions
`app.py` uses, so headless and interactive runs are directly comparable —
they just skip the frame-by-frame rendering and let you regenerate the whole
evidence set in one deterministic pass instead of manually driving the app
through all 7 levels.

---

## 4. Gridworld rules (implemented in `env.py`)

- Actions: up / down / left / right. Moving into a rock or off the grid is a
  no-op (agent stays put).
- Stepping into fire, or into a tile a monster occupies, is instant death —
  checked immediately after the agent's own move, before any reward.
- Apples: +1 once, on pickup.
- Keys: no reward, but required (consumed) to open a chest.
- Chests: +2, once, and only if the agent is currently holding a key.
- After the agent moves (and reward/pickup is resolved), each monster
  independently has a 40% chance to take one uniformly-random step among its
  currently-legal moves (`GridWorld.monster_step`); if that step lands it on
  the agent, the agent dies.
- Episode ends when every apple is collected, every key is collected, and
  every chest is opened — or on death.

## 5. Levels

| Level | Contents                                                 | Task                         |
| ----- | -------------------------------------------------------- | ---------------------------- |
| 0     | Apples only, straight corridor                           | Task 1 — Q-learning          |
| 1     | One apple behind two rows of fire                        | Task 2 — SARSA vs Q-learning |
| 2, 3  | Multiple apples, a key, a chest, rocks (3 also has fire) | Task 3                       |
| 4     | One monster                                              | Task 4                       |
| 5     | Two monsters                                             | Task 4                       |
| 6     | Apples, key, chest, no hazards                           | Task 5 — intrinsic reward    |

---

## 6. Evidence and results (from `training_curves/summary.csv`, 1200 episodes each)

### Task 1 — Level 0, Q-learning shortest path (B5)

`training_curves/plots/paths/level0_q_extrinsic.png` shows a greedy
(epsilon=0) rollout after training: the agent walks directly to column 8 and
straight down it, collecting all 6 apples in **15 steps** — the shortest
possible path — with a 96% success rate and 0 deaths across training
(`training_curves/plots/level0_q_vs_sarsa.png` for the full learning curve).

### Task 2 — Level 1, SARSA vs Q-learning near hazards (C3)

Level 1's only apple sits at the far end of a corridor flanked top and bottom
by fire, so travelling that corridor keeps the agent adjacent to fire the
entire way — a direct analogue of Sutton & Barto's "cliff walking" example.

`training_curves/plots/level1_q_vs_sarsa_path.png` overlays the two learned
greedy paths: **Q-learning** (off-policy, assumes the _next_ action is chosen
optimally) walks straight down the corridor, immediately alongside fire the
whole way. **SARSA** (on-policy, its value estimate reflects the exploratory
policy actually used during training, including epsilon-random moves that
can wander into fire) instead detours away from the corridor for part of the
route, trading a couple of extra steps for a larger safety margin from the
hazard — visibly "more conservative around hazards" as expected.

This also shows up in the raw numbers: because both policies still take
epsilon-random actions during training and fire is one misstep away for the
whole corridor, Q-learning racks up more fire deaths over the full 1200
episodes (532) than SARSA (588 — comparable, since both explore with the same
schedule) but SARSA's on-policy value estimates converge to a policy that
gives fire a wider berth, at the cost of ~2 extra steps per episode once
converged (avg. successful-episode length 22.3 for Q vs 24.2 for SARSA).

### Task 3 — Levels 2-3, multiple apples + key + chest

Both algorithms complete the full apple/key/chest sequence correctly on both
layouts (92%/87% success on Level 2, 65%/66% on Level 3 — Level 3 is harder
because it also contains fire). See
`training_curves/plots/level2_q_vs_sarsa.png` and `level3_q_vs_sarsa.png`.

### Task 4 — Levels 4-5, stochastic monsters

Monsters move with probability `monsterMoveProbability` (0.4) after every
agent action, so the transition model is genuinely stochastic. Both
algorithms learn partial avoidance (Level 4: 63%/48% success, mostly ended by
monster collisions; Level 5, with two monsters: 48%/39% success) — see
`training_curves/plots/level4_q_vs_sarsa.png` and `level5_q_vs_sarsa.png`
for the learning curves.

### Task 5 — Level 6, intrinsic exploration reward (F)

Intrinsic reward is `r_i = intrinsicRewardStrength / sqrt(n(s) + 1)`, added
to (not replacing) the environment reward, using a per-episode visit counter
`n(s)`; environment rewards themselves are unchanged
(`training_curves/level6_*_extrinsic.csv` vs the environment-return column
in `..._intrinsic.csv` — see `metrics.py`'s `env_return_history`, which is
tracked separately from the intrinsic-inclusive `total_return`).

**Observed result: at the shipped `intrinsicRewardStrength = 0.2`, intrinsic
reward _hurts_ Level 6 rather than helping** —
`training_curves/plots/level6_q_intrinsic_vs_extrinsic.png` shows Q-learning
without intrinsic reward converging to a perfect return of 5.0 by ~episode
150 and staying there, while _with_ intrinsic reward it is noisier, takes
longer to reach 5.0, and then visibly collapses back down to ~4.2-4.5 for the
remainder of training (SARSA shows the same collapse-then-partial-recovery
pattern — `level6_sarsa_intrinsic_vs_extrinsic.png`).

**Why:** unlike a potential-based shaping term (Ng, Harada & Russell 1999),
this bonus does not telescope to zero over a trajectory and is not
policy-invariant — it permanently rewards visiting a less-common state within
the _current_ episode, even once the agent already knows the optimal route,
because `n(s)` resets every episode. At strength 0.2 (20% of an apple's
reward, 10% of the chest's), that bonus is large enough to compete with
actually finishing the level, so once epsilon has decayed and the greedy
component of the policy dominates, the agent settles into a
locally-optimal detour-and-explore habit instead of the shortest path.

`tune_intrinsic.py` confirms this is a scale problem, not a correctness
problem — sweeping the strength on Level 6 (`training_curves/intrinsic_strength_sweep.csv`):

| Strength       | Q success rate | SARSA success rate |
| -------------- | -------------- | ------------------ |
| 0.0 (off)      | 0.96           | 0.95               |
| 0.02           | 0.95           | 0.86               |
| 0.05           | 0.94           | 0.82               |
| 0.10           | 0.94           | 0.74               |
| 0.20 (shipped) | 0.52           | 0.66               |
| 0.40           | 0.51           | 0.32               |

Success rate degrades smoothly as the bonus grows relative to the
environment reward, with SARSA more sensitive at small-to-medium strengths
(its on-policy updates fold the transient per-episode bonus into the value
function more directly) and both algorithms collapsing once the bonus
exceeds roughly 5-10% of an apple's reward. A strength around 0.02-0.05 would
give Level 6 the exploration benefit intrinsic reward is meant to provide
without this degradation — Level 6's layout has no hazards and is small
enough that epsilon-greedy alone already explores it fully, so there was
never a hard-exploration problem for the bonus to solve, and the shipped
value overcorrects for it.

---

## 7. Config (`config.json`)

Shared by every level (the assignment only requires hyperparameters to come
from _a_ config file, not one per level):

```json
{
  "episodes": 1200,
  "alpha": 0.2,
  "gamma": 0.95,
  "epsilonStart": 1.0,
  "epsilonEnd": 0.05,
  "epsilonDecayEpisodes": 700,
  "maxStepsPerEpisode": 400,
  "monsterMoveProbability": 0.4,
  "intrinsicRewardStrength": 0.2
}
```

(plus rendering/speed options: `fpsVisual`, `fpsFast`, `rapidStepsPerFrame`,
`rapidRenderEvery`, `tileSize`, `panelWidth`, `seed`.)
