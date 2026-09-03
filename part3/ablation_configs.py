"""
part3/ablation_configs.py
===================================
Part III (Masters, Option A — Ablation study).

Design: one BASELINE config + three single-factor VARIANTS, each flipping
exactly one axis away from baseline. This is standard ablation methodology
(change one factor at a time so any performance delta is attributable to
that factor) and covers all three axes suggested in the architecture doc:

  axis "shaping"     : potential-based reward shaping on vs off
  axis "observation"  : full 15-dim observation vs spawner-feature ablated
  axis "network"       : [64,64] vs [128,128] policy/value MLP

Control scheme is held fixed at "direct" across the whole study (a second
free variable would turn this into a combinatorial sweep and blur which
factor caused which effect -- out of scope for a single-factor ablation).

Changed from "rotate" to "direct" on 2026-08-27: a real 1M-timestep run
showed "rotate" essentially never breaks out of a ~-22 task-reward
plateau (PPO struggles to learn the turn+thrust aiming problem in this
budget), while "direct" reliably breaks through by ~700-800k steps. Using
"rotate" as the ablation's base scheme would have measured "can this
variant learn at all" (mostly: no) rather than the intended per-axis
effect -- "direct" gives each variant a fair chance to demonstrate a
signal within a practical timestep budget.
"""

BASELINE = dict(
    name="baseline",
    axis="baseline",
    control_scheme="direct",
    shaping_enabled=True,
    include_spawner_feature=True,
    net_arch=(64, 64),
    lr=3e-4,
)

ABLATION_CONFIGS = [
    BASELINE,
    dict(BASELINE, name="no_shaping", axis="shaping", shaping_enabled=False),
    dict(BASELINE, name="no_spawner_feature", axis="observation", include_spawner_feature=False),
    dict(BASELINE, name="bigger_network", axis="network", net_arch=(128, 128)),
]
