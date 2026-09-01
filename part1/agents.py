"""Tabular Q-learning and SARSA agents: Q-table, action selection, updates."""
import random
from typing import Dict, Tuple

from env import ALL_ACTIONS


def linear_epsilon(ep, start, end, decay_ep):
    if decay_ep <= 0:
        return end
    t = min(ep / decay_ep, 1.0)
    return start + t * (end - start)


class QTable:
    def __init__(self):
        self.q: Dict[Tuple[tuple, int], float] = {}

    def get(self, s, a):
        return self.q.get((s, a), 0.0)

    def set(self, s, a, v):
        self.q[(s, a)] = v

    def best_value(self, s):
        return max(self.get(s, a) for a in ALL_ACTIONS)

    def best_actions(self, s):
        vals = [self.get(s, a) for a in ALL_ACTIONS]
        m = max(vals)
        return [a for a, v in zip(ALL_ACTIONS, vals) if v == m]


def epsilon_greedy(qtab: QTable, s, eps):
    if random.random() < eps:
        return random.choice(ALL_ACTIONS)
    return random.choice(qtab.best_actions(s))


def greedy_action(qtab: QTable, s):
    """Pure greedy action with random tie-breaking (used for evaluation rollouts)."""
    return random.choice(qtab.best_actions(s))


def q_learning_update(qtab: QTable, s, a, r, sp, alpha, gamma, done=False):
    """Off-policy update: bootstraps from the max over next-state actions."""
    current = qtab.get(s, a)
    target = r if done else r + gamma * qtab.best_value(sp)
    qtab.set(s, a, current + alpha * (target - current))


def sarsa_update(qtab: QTable, s, a, r, sp, ap, alpha, gamma, done=False):
    """On-policy update: bootstraps from the action actually chosen next."""
    current = qtab.get(s, a)
    target = r if done else r + gamma * qtab.get(sp, ap)
    qtab.set(s, a, current + alpha * (target - current))
