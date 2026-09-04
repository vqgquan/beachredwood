# Part III — Ablation Study (Masters, Option A)

Which of Part II's design decisions actually matter? One baseline plus three
single-factor variants, each flipping exactly one decision with everything
else held identical.

---

## 1. `arena_env.py` is not modified

Every variant is a **subclass** of `part2.arena_env.ArenaEnv`:

```python
class NoShaping(ArenaEnv):   R_SHAPE = 0.0
class NoStepCost(ArenaEnv):  R_STEP  = 0.0
```

This is the stronger experimental design, not just the tidier one. The file
under test keeps a zero-line diff, so the baseline arm of every comparison is
provably the same code that produced Part II's existing results. Adding
`if ablation_enabled:` branches inside `ArenaEnv` would mean the baseline arm
runs through code paths that did not exist when those results were produced.

It works because `arena_env.py` exposes every reward weight as a class
attribute and builds the observation in an overridable method.

---

## 2. Files

| File | Role |
|---|---|
| `ablation_envs.py` | The three subclasses. No RL, no training. |
| `ablation_configs.py` | Pure data: which variant, which axis, how it is scored. |
| `run_ablation.py` | Trains and scores each arm, writes `results/ablation_results.csv`. |
| `plot_ablation.py` | Turns the CSV into the report figures. |
| `seed_check.py` | Re-runs contested arms across seeds — see section 5. |

---

## 3. The three axes

| Variant | Changes | Question it answers |
|---|---|---|
| `no_shaping` | `R_SHAPE` 3.0 → 0.0 | Can the agent find the objective from sparse kill/phase rewards alone? |
| `no_step_cost` | `R_STEP` −0.02 → 0.0 | Does the time cost actually stop the agent hiding, as the comment in `arena_env.py` claims? |
| `no_ship_frame` | `obs[9,10,15,16]` → 0 | Does the pre-computed "am I aimed at it" signal matter, or can the agent derive it from world-frame direction and its own heading? |

All four arms use the rotation scheme. Rotation is the scheme Part II ships a
trained model for, and it is the one where aiming is a genuine control problem
— under the direct scheme an agent can close on a target without ever solving
"am I pointing at it", which would make the third axis meaningless.

---

## 4. How the arms are scored

**Best checkpoint, not the final one.** PPO here can plateau, break through,
and regress again within a single run. An `EvalCallback` tracks the best
policy during training and that is what gets scored.

**A common reward scale.** Reward ablations train on their ablated reward but
are scored on the untouched `ArenaEnv`. Without this, `no_step_cost` would be
graded on a scale that never charges it for time and would beat the baseline
for free. The observation ablation is scored on its own observation, because
that vector is part of that agent's interface — feeding it inputs it never saw
in training would measure nothing. Its reward is untouched either way.

**Behaviour, not just reward.** Episode length, kills and spawners destroyed
are recorded alongside reward, because two agents can score alike for opposite
reasons: one fighting well, one refusing to fight at all.

---

## 5. Seeds

Two sources of noise sit under every number here, and both are measured.

**Between seeds.** `run_ablation.py` runs one seed. That is enough to produce a
table and not enough to draw a conclusion from it — PPO on this task can break through or
fail to within the same budget depending only on initialisation, so a
single-seed gap between two arms may be noise.

`seed_check.py` re-runs the contested arms across several seeds. Read its
output before writing any sentence of the form "X is better than Y".

**Within one seed.** Training the seed-0 arm twice, with identical code and
configuration, produced results differing by up to 9.7 reward (24% relative on
`no_step_cost`). Eight parallel environments and multi-threaded PyTorch make the
sample order non-deterministic, and small gradient differences compound over
700k steps. A fixed seed therefore makes a run approximately repeatable, not
bit-identical. See `results/reproducibility_note.txt` for the measurements.

---

## 6. Running it

```bash
python part3/run_ablation.py --timesteps 700000 --n-envs 8
python part3/plot_ablation.py
python part3/seed_check.py --seeds 1 2
```

Outputs land in `part3/results/`.
