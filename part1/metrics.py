"""Per-run training metrics and CSV export, shared by app.py and generate_evidence.py."""
import csv
import os


def make_metrics():
    return {
        "episodes": 0,
        "successes": 0,
        "deaths": 0,
        "timeouts": 0,
        "fire_deaths": 0,
        "monster_deaths": 0,
        "steps_total": 0,
        "successful_steps": 0,
        "return_total": 0.0,
        "intrinsic_total": 0.0,
        "return_history": [],
        "env_return_history": [],
        "status_history": [],
    }


def record_episode(metrics, steps, total_reward, intrinsic_episode, env_reward_episode,
                    done, alive, info):
    metrics["episodes"] += 1
    metrics["steps_total"] += steps
    metrics["return_total"] += total_reward
    metrics["intrinsic_total"] += intrinsic_episode
    metrics["return_history"].append(total_reward)
    metrics["env_return_history"].append(env_reward_episode)

    if done and alive:
        metrics["status_history"].append("success")
        metrics["successes"] += 1
        metrics["successful_steps"] += steps
    elif done:
        metrics["status_history"].append("death")
        metrics["deaths"] += 1
        if info.get("event") == "fire_death":
            metrics["fire_deaths"] += 1
        elif info.get("event") in ("monster_death", "monster_collision"):
            metrics["monster_deaths"] += 1
    else:
        metrics["status_history"].append("timeout")
        metrics["timeouts"] += 1


def save_training_curve(level, algorithm, metrics, intrinsic_enabled, out_dir=None):
    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "training_curves")
    os.makedirs(out_dir, exist_ok=True)
    intrinsic_label = "intrinsic" if intrinsic_enabled else "extrinsic"
    path = os.path.join(out_dir, f"level{level}_{algorithm}_{intrinsic_label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "env_return", "total_return", "status"])
        for episode, (env_value, total_value, status) in enumerate(
            zip(metrics["env_return_history"], metrics["return_history"],
                metrics["status_history"]), start=1
        ):
            writer.writerow([episode, env_value, total_value, status])
    return path
