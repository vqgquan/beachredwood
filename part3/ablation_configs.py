"""
part3/ablation_configs.py

Part III (Masters, Option A). One BASELINE plus three single-factor variants:
each flips exactly one design decision, everything else held identical, same
seed, same timestep budget -- so any performance gap is attributable to that
factor alone.

Control scheme is fixed at "rotation" across the whole study. Two reasons:
it is the scheme part2 already ships a trained model for, and it is the scheme
where the third axis (ship-frame aiming) is a real control problem rather than
a convenience -- under the direct scheme the agent can close on a target
without ever solving "am I pointing at it".

The three axes cover reward design (two) and observation design (one), which
is where this project's design decisions actually live.
"""

BASELINE = dict(
    name="baseline",
    axis="baseline",
    env="ArenaEnv",
    scheme="rotation",
    note="The configuration part2 ships.",
)

ABLATION_CONFIGS = [
    BASELINE,
    dict(BASELINE, name="no_shaping", axis="reward: shaping", env="NoShaping",
         note="Potential-based pull toward the nearest spawner removed. Tests "
              "whether the agent can find the objective from the sparse "
              "kill/phase rewards alone."),
    dict(BASELINE, name="no_step_cost", axis="reward: time cost", env="NoStepCost",
         note="Per-decision time cost removed. Tests the comment in "
              "arena_env.py that hiding for a whole episode must not pay."),
    dict(BASELINE, name="no_ship_frame", axis="observation", env="NoShipFrame",
         note="Target direction in the ship's own frame zeroed. World-frame "
              "direction is kept, so the agent still knows where the target "
              "is -- it loses only the ready-made aiming check."),
]

# Reward ablations are scored on the untouched ArenaEnv so every arm is graded
# on the same reward scale. The observation ablation is scored on its own
# observation, because that vector is part of the agent's interface.
EVAL_ON_BASELINE_ENV = {"baseline", "no_shaping", "no_step_cost"}
