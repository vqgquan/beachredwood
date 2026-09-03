# Part II + Part III — Arena Deep RL

Implements the architecture agreed in `the architecture agreed at the start of Part II`:
game logic (`ArenaSimulation`) is fully decoupled from the RL adapter
(`ArenaEnv`), PPO trains headless, evaluation renders.

## Setup

```bash
pip install -r requirements.txt
```

## Part II — file map

| File | Role |
|---|---|
| `arena_simulation.py` | Pure game logic + Pygame rendering. Player/Enemy/Spawner/Bullet, phases, collisions. **No RL concepts.** |
| `arena_env.py` | `ArenaEnv(gymnasium.Env)`. Action/observation spaces, reward + potential-based shaping. **The only RL-aware module.** |
| `training_common.py` | Shared PPO/vec-env/callback construction used by both training scripts and the ablation runner. |
| `train_control1.py` | Trains PPO on the **rotate** scheme → `models/ppo_rotate.zip` |
| `train_control2.py` | Trains PPO on the **direct** scheme → `models/ppo_direct.zip` |
| `eval_control1.py` / `eval_control2.py` | Loads the best checkpoint, `render_mode="human"`, prints per-episode outcome. Use these to record the demo video. |
| `tests/test_smoke.py` | 15 unit tests: physics correctness, event firing, Gymnasium `check_env` compliance for both control schemes, ablation-relevant behaviours. |

### Run it

```bash
# 1. sanity check everything still works on your machine
python -m pytest tests/test_smoke.py -v

# 2. train (headless; SDL_VIDEODRIVER=dummy is set automatically)
python train_control1.py --timesteps 500000 --n-envs 8
python train_control2.py --timesteps 500000 --n-envs 8

# 3. watch it play / record the demo video
python eval_control1.py --model models/best/rotate/best_model.zip --episodes 5
python eval_control2.py --model models/best/direct/best_model.zip --episodes 5
```

`--timesteps 500000` is a starting point, not a target — watch
`ep_rew_mean` in the training log; once it plateaus you have enough.
On a laptop CPU, 500k steps with `n_envs=8` typically takes on the order
of 10-20 minutes for this observation size (no images, 15-dim vector).

**Before recording the video**, double-check the `--model` path you eval
is the exact `.zip` you are submitting — mismatched video/submission model
is explicitly called out as a common way to lose marks.

## Real training results (2026-08-27)

`models/`, `logs/`, and `../part3/results/` in this folder are now
from **real training runs** (1M timesteps per control scheme, 800k
timesteps per ablation config — not the short smoke tests the earlier
version of this README warned about). Two real bugs were found and fixed
while producing these:

1. **`EvalCallback` was scoring policies WITH shaping still on.**
   `training_common.build_vec_envs` built the eval env with the same
   `shaping_enabled` as the training env. That made the reported "eval
   reward" measure "task reward + proximity-to-spawner bonus", not the
   task alone — a 500k-step `direct` model that reported **+3.9** eval
   reward actually scored **-23** once evaluated on task-only reward
   (shaping off). Fixed: `build_vec_envs` now defaults the eval env to
   `eval_shaping_enabled=False`, independent of the training env's
   setting, matching what `../part3/run_ablation.py` already did
   for its own fair-comparison eval. **If you retrain, `EvalCallback`'s
   numbers and `best_model.zip` selection are now task-only and trustworthy
   — no change needed on your end, just be aware the numbers you see now
   read lower (more honest) than an unfixed run would report.**
2. **`run_ablation.py` scored only the FINAL model**, with no
   best-checkpoint tracking. PPO on this task turns out to have long
   plateaus followed by a breakthrough, sometimes followed by a late
   collapse (see below) — evaluating only the last step risks scoring an
   unlucky collapsed snapshot. Fixed: `run_one_config` now runs an
   `EvalCallback` during each config's training and evaluates the saved
   best checkpoint, same mechanism `train_control1.py`/`train_control2.py`
   already used.

**Rotate vs direct, for real:** `direct` (4-way move + shoot) reliably
breaks through to a working policy around 700-800k timesteps; `rotate`
(turn + thrust + shoot) does **not** break through within 1M timesteps —
task-only reward stayed flat at roughly -21 to -23 the entire run (best
checkpoint: -22.1). `direct`'s best checkpoint (step 740k) scored **-1.6**
task-only reward over 30 eval episodes — a large, genuine gap. This is
strong, real evidence for the report's control-scheme comparison (R5):
aiming via rotation is a substantially harder credit-assignment problem
for PPO than direct 4-way movement, at a comparable compute budget.
`ablation_configs.py` was switched from `control_scheme="rotate"` to
`"direct"` for this reason — an ablation run on a scheme that mostly
fails to learn at all would only show "can this variant learn" (no),
not the intended per-axis effect.

**PPO instability, worth a sentence in the report:** the `direct` run's
`ep_rew_mean` did not monotonically improve — after the ~700k-step
breakthrough it kept climbing for a while, then regressed sharply near
the end of training (final-step policy was back to ~-22, no better than
the pre-breakthrough plateau). This is why the `EvalCallback`
best-checkpoint mechanism (see bug #2 above) matters: `models/best/direct/best_model.zip`
is the good policy from step ~740k, not whatever the final step happened
to land on. Always eval/demo from `models/best/{scheme}/best_model.zip`,
never `models/ppo_{scheme}.zip` (the final-step model) — `eval_control1.py`/
`eval_control2.py` already default to the `best/` path for this reason.

**Ablation results** (`direct` scheme, 800k timesteps/config, task-only
reward, mean ± std over 20 eval episodes from the best checkpoint):

| Config | axis | mean_reward | Interpretation |
|---|---|---|---|
| `baseline` (shaping on, full obs, [64,64]) | — | -14.05 ± 9.36 | Breaking through, but late and high-variance (only reached ~-12 in the last few eval points) |
| `no_shaping` | shaping | -22.38 ± 0.70 | **Never breaks through** — stayed on the pre-breakthrough plateau for the full 800k |
| `no_spawner_feature` | observation | -22.43 ± 0.24 | **Also never breaks through** — without seeing the spawner, the agent can't act on the shaping signal even though shaping is still computed |
| `bigger_network` | network | **-4.65 ± 0.39** | Clearly the best — broke through earliest (~700k) and reached full episode length (1801 steps = surviving the whole episode) |

Read together, shaping and the spawner observation feature look
*necessary* for this task to be learnable at all within this budget (remove
either and the agent never breaks through), while network capacity is
what determines *how well* it does once it can learn — a genuinely
interesting, non-obvious finding worth building the Part III write-up
around, not just "bigger number = better".

### Design choices worth putting in the report

- **Observation is a fixed 15-dim vector**, never pixels — this is what
  makes `Box(shape=(15,))` valid for a `MlpPolicy` and keeps training fast.
  Sentinel values (`dx=dy=0, dist=1.0`) are used when no enemy/spawner
  exists, so the shape never changes even when the arena is briefly empty
  — this directly avoids the "observation không fixed-size" pitfall listed
  in the risk section of the architecture doc.
- **Reward shaping is potential-based** (`Φ(s) = -k·dist(player, nearest
  active spawner)`, `r_shape = γΦ(s') - Φ(s)`). Per Ng, Harada & Russell
  (1999) this provably does not change the optimal policy, unlike naive
  distance-delta rewards, and it's the reason `shaping_enabled` and
  `shaping_k` are constructor args rather than baked in.
- **One action per Discrete step** (e.g. you either turn or thrust, not
  both, in the `rotate` scheme). This is a deliberate simplification of a
  richer `MultiDiscrete` action space — worth a sentence in the report
  acknowledging the trade-off (simpler policy head vs. slightly less
  expressive control).
- **Reward-hacking guard**: bullet-hit reward (+1.0) only fires on an
  actual enemy HP-depleting collision event from `ArenaSimulation`, never
  from proximity — so the shaping term alone cannot be farmed without
  ever engaging (per the "reward hacking" risk in the doc, an agent that
  just orbits near a spawner without shooting gets shaping deltas near
  zero once it stops closing distance, and zero task reward).

## Part III — Option A: Ablation study (Masters, 4 pts)

`../part3/` runs a single-factor ablation: one **baseline** PPO
config vs. three **variants**, each flipping exactly one axis so any
performance delta is attributable to that factor alone:

| Axis | Baseline | Variant |
|---|---|---|
| Reward shaping | on | off |
| Observation | full (15-dim) | spawner-feature ablated (dims 9-11 zeroed) |
| Network capacity | `[64, 64]` | `[128, 128]` |

```bash
python ../part3/run_ablation.py --timesteps 800000 --n-envs 8
python ../part3/plot_ablation.py
```

Each config takes ~5-6 minutes at 800k timesteps (measured in-session). Use
`--only <name>` (e.g. `--only bigger_network`) to (re-)run a single config
without retraining the other three — results merge into the existing CSV.

Outputs land in `../part3/results/`:
- `ablation_results.csv` — raw numbers (mean/std reward, train time)
- `ablation_shaping.png`, `ablation_observation.png`, `ablation_network.png`
  — bar charts, baseline vs. variant, for the report
- `ablation_summary.txt` — plain-text table

Evaluation for every config runs with **shaping forced off**
(`shaping_enabled=False` in the eval env only, never in training), so the
`shaping_on` vs `shaping_off` comparison is scored on the same
task-reward scale rather than one side getting "free" shaping points in
its own eval.

`--timesteps 800000` was needed for `baseline`/`bigger_network` to reach
their breakthrough (see "Real training results" above) — a shorter budget
(150k, the original default) is not enough for this task to show a signal
at all; every config just looks like a flat, uninformative plateau.

### Extending the ablation (optional)

The doc also lists "lr cao/thấp" as an alternative to the network-size
axis. `training_common.build_ppo(net_arch=..., learning_rate=...)`
already exposes both knobs — add a config to `ablation_configs.py` like:

```python
dict(BASELINE, name="lr_high", axis="learning_rate", lr=1e-3)
```

and it will flow through `run_ablation.py` / `plot_ablation.py`
automatically (the plot titles map is keyed by axis name — add a title
for `"learning_rate"` in `AXIS_TITLES` if you do this).

## What's still open

- Report (PDF ≤ 10 pages) and demo video — not started. The "Design
  choices worth putting in the report" and "Real training results"
  sections above are meant to seed that write-up.
- **Demo video**: `rotate`'s best checkpoint still performs poorly
  (task-only -22) — record the video with `eval_control2.py` (direct
  scheme, best checkpoint -1.6) as the strong demo, and consider whether
  to also show `rotate` honestly as "still learning" rather than pretend
  it works well, or invest more training time on it first (it may just
  need a longer/curriculum-style budget past 1M steps — it never showed
  a breakthrough signal even at 1M, unlike `direct` at ~750k).
- `tests/test_smoke.py` (15 tests) verified passing during this session.
- `pip install -r requirements.txt` still needed on whatever machine you
  run these scripts from if you haven't already (this session ran them in
  its own sandbox — confirm your local environment matches before
  re-running anything).
