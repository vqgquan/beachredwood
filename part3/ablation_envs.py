"""
part3/ablation_envs.py

Ablated variants of part2's ArenaEnv, built by subclassing.

arena_env.py is NOT modified. That is deliberate, and it is the stronger
experimental design rather than merely the politer one: the file under test
keeps a zero-line diff, so the baseline arm of every comparison is provably
the same code that produced part2's existing results. Adding
`if ablation_enabled:` branches inside ArenaEnv would mean the baseline arm
now runs through code paths that did not exist when those results were
produced.

This works because part2/arena_env.py exposes every reward weight as a class
attribute and builds the observation in an overridable method.

Three axes, one design decision each:

  NoShaping     R_SHAPE  3.0 -> 0.0   potential-based pull toward the spawner
  NoStepCost    R_STEP  -0.02 -> 0.0  the per-decision time cost
  NoShipFrame   obs[9,10,15,16] -> 0  target direction in the ship's own frame

Scoring, and why it differs per axis:

  Reward ablations are TRAINED on the ablated reward but SCORED on the
  unmodified ArenaEnv. Otherwise NoStepCost would be graded on a scale that
  simply does not charge it for time, and would beat the baseline for free.

  The observation ablation is trained AND scored on its own observation: the
  ablated vector is part of that agent's interface, and feeding it inputs it
  never saw during training would measure nothing. Its reward is untouched,
  so its score is already on the baseline scale.
"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "part2"))

from arena_env import ArenaEnv  # noqa: E402

# Indices of the target direction expressed in the ship's own frame:
# 9-10 nearest enemy, 15-16 nearest spawner. The world-frame directions
# (7-8, 13-14) are left intact, so the agent still knows WHERE the target is
# -- it just loses the pre-computed "am I pointing at it" signal.
SHIP_FRAME_DIMS = (9, 10, 15, 16)


class NoShaping(ArenaEnv):
    """Potential-based shaping toward the nearest spawner removed."""
    R_SHAPE = 0.0


class NoStepCost(ArenaEnv):
    """Per-decision time cost removed.

    Prediction: with no cost to existing, surviving while doing nothing scores
    0, which beats engaging imperfectly. If the agent discovers that, episode
    length should rise toward the cap while kills fall toward zero.
    """
    R_STEP = 0.0


class NoShipFrame(ArenaEnv):
    """Ship-frame target directions zeroed out of the observation.

    The agent keeps world-frame direction, so it still knows where the target
    is; what it loses is the ready-made check for whether it is aimed at it.
    Under the rotation scheme, aiming is the whole control problem.
    """

    def _get_obs(self):
        obs = super()._get_obs()
        for i in SHIP_FRAME_DIMS:
            obs[i] = 0.0
        return obs


# Keyed by class name so ablation_configs.py can name them as strings.
ABLATION_ENVS = {
    "NoShaping": NoShaping,
    "NoStepCost": NoStepCost,
    "NoShipFrame": NoShipFrame,
}
