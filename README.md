# Assessment 3 — Reinforcement Learning

RMIT · Machine Learning & AI · Team: Khang, Quan, Ha

Classical tabular RL in a Pygame gridworld, deep RL in a real-time Pygame
arena, and an ablation study on the arena agent.

| Folder   | Contents                                                        | Assessment part      |
| -------- | --------------------------------------------------------------- | -------------------- |
| `part1/` | Pygame gridworld, tabular Q-learning + SARSA, 7 levels            | Part I, Tasks 1–5    |
| `part2/` | Real-time Pygame arena, Gymnasium env, PPO on two control schemes | Part II              |
| `part3/` | Single-factor ablation study on the Part II agent                 | Part III — Option A  |

Each folder has its own README with a file-by-file breakdown.

---

## Setup

One virtual environment is shared by all three parts. Python **3.13** —
pygame has no prebuilt wheel for 3.14+ on macOS and will fail to compile.

```bash
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r part1/requirements.txt
pip install -r part2/requirements.txt
```

`part3` imports from `part2`; no separate install is needed.

---

## Running each part

**Part I — interactive gridworld (the graded deliverable)**

```bash
cd part1 && python app.py
```

Keys `1`–`7` pick a level, `Q`/`S` pick the algorithm, `T` speeds training up,
`V` toggles rendering, `P` pauses evaluation.

**Part I — regenerate all report evidence, headless**

```bash
cd part1
python generate_evidence.py     # all 7 levels x 2 algorithms x intrinsic on/off
python plot_curves.py           # renders the PNGs used in the report
```

**Part II — train and watch the arena agents**

```bash
cd part2
python train_control1.py --timesteps 1000000 --n-envs 8    # rotation scheme
python train_control2.py --timesteps 1000000 --n-envs 8    # direct scheme

python eval_control1.py --model models/best/rotate/best_model.zip --episodes 5
python eval_control2.py --model models/best/direct/best_model.zip --episodes 5
```

Always evaluate and record from `models/best/<scheme>/best_model.zip`, not
`models/ppo_<scheme>.zip`. PPO on this task plateaus, breaks through, and can
then regress, so the final-step model is often worse than the best checkpoint.

**Part III — ablation study**

```bash
python part3/run_ablation.py --timesteps 800000 --n-envs 8
python part3/plot_ablation.py
```

---

## Where the report evidence lives

| Evidence                                  | Path                                    |
| ----------------------------------------- | --------------------------------------- |
| Part I learning curves (CSV)              | `part1/training_curves/`                |
| Part I greedy rollout paths               | `part1/training_curves/paths/`          |
| Part I rendered figures                   | `part1/training_curves/plots/`          |
| Part I run summary table                  | `part1/training_curves/summary.csv`     |
| Part II TensorBoard logs                  | `part2/logs/tensorboard/`               |
| Part II trained models                    | `part2/models/`                         |
| Part III results, plots and summary       | `part3/results/`                        |

---

## Tests

```bash
python -m pytest part2/tests/test_smoke.py -v
```

15 tests covering arena physics, event firing, and Gymnasium `check_env`
compliance for both control schemes.
