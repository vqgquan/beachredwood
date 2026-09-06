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
attack spawners. DQN is also supported
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

A GPU does not meaningfully speed up PPO in this project (the policy network is
too small); `--n-envs`/`--subproc` is the effective speedup for PPO, while
`--device cuda` mainly benefits DQN.

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

## 4. Notes

* Never pass `render_mode="human"` during training — rendering caps the loop at
  30 FPS and makes training roughly 100× slower.
* Checkpoints are written to `models/checkpoints/` every 50k steps, so an
  interrupted run is not lost.
* Every random draw goes through `self.np_random`, so `reset(seed=n)` reproduces
  exactly the same arena layout.
