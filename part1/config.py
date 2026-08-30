"""Shared configuration for the Part I gridworld.

Loads part1/config.json (falling back to sane defaults) and seeds the RNG once
at import time, so every entry point (the interactive app, the headless
evidence generator) sees identical hyperparameters and reproducible runs.
"""
import json
import os
import random

DEFAULT_CFG = {
    "episodes": 800,
    "alpha": 0.2,
    "gamma": 0.95,
    "epsilonStart": 1.0,
    "epsilonEnd": 0.05,
    "epsilonDecayEpisodes": 700,
    "maxStepsPerEpisode": 400,
    "fpsVisual": 30,
    "fpsFast": 1000,
    "rapidStepsPerFrame": 25,
    "rapidRenderEvery": 20,
    "monsterMoveProbability": 0.4,
    "intrinsicRewardStrength": 0.2,
    "tileSize": 48,
    "panelWidth": 320,
    "seed": 42,
}


def load_config():
    cfg = DEFAULT_CFG.copy()
    path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


CFG = load_config()

EPISODES = int(CFG["episodes"])
ALPHA = float(CFG["alpha"])
GAMMA = float(CFG["gamma"])
EPS_START = float(CFG["epsilonStart"])
EPS_END = float(CFG["epsilonEnd"])
EPS_DECAY_EP = int(CFG["epsilonDecayEpisodes"])
MAX_STEPS = int(CFG["maxStepsPerEpisode"])
FPS_VISUAL = int(CFG["fpsVisual"])
FPS_FAST = int(CFG["fpsFast"])
RAPID_STEPS = int(CFG["rapidStepsPerFrame"])
RAPID_RENDER_EVERY = int(CFG["rapidRenderEvery"])
MONSTER_MOVE_PROBABILITY = float(CFG["monsterMoveProbability"])
INTRINSIC_REWARD_STRENGTH = float(CFG["intrinsicRewardStrength"])
STEP_PENALTY = float(CFG.get("stepPenalty", 0.0))
TILE_SIZE = int(CFG["tileSize"])
PANEL_W = int(CFG["panelWidth"])
SEED = int(CFG["seed"])

random.seed(SEED)
