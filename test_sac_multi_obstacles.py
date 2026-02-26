#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import csv
import numpy as np
from stable_baselines3 import SAC
from biped_env_obs3 import BipedWalkingEnv


# Resolve model path relative to this file so it works regardless of CWD
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models_obs3", "sac_biped_goal.zip")


def _safe_get_episode_metrics(env, info):
    """Try multiple ways to get episode metrics."""
    if hasattr(env, "get_episode_metrics"):
        try:
            return env.get_episode_metrics()
        except Exception:
            pass
    if isinstance(info, dict) and "episode_metrics" in info:
        return info["episode_metrics"]
    return None


def test_sac_with_obstacles(
    num_obstacles: int,
    model_path: str = DEFAULT_MODEL_PATH,
    episodes: int = 100,
    render: bool = False,
    deterministic: bool = True,
    save_csv: bool = True,
):
    """
    Test SAC model with a specific number of obstacles.
    
    Args:
        num_obstacles: Number of obstacles to spawn in the environment
        model_path: Path to the trained SAC model
        episodes: Number of episodes to evaluate
        render: Whether to render the environment (GUI)
        deterministic: Whether to use deterministic actions
        save_csv: Whether to save results to CSV
    """
    # Create environment
    env = BipedWalkingEnv(render=render)
    
    # Override the number of obstacles
    env.num_obstacles = num_obstacles
    
    print(f"\n{'='*70}")
    print(f"Testing with {num_obstacles} obstacles")
    print(f"Episodes: {episodes}, Render: {render}, Deterministic: {deterministic}")
    print(f"{'='*70}\n")

    # Load the model
    if not os.path.isfile(model_path):
        available = []
        models_dir = os.path.join(os.path.dirname(__file__), "models_obs3")
        if os.path.isdir(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith(".zip"):
                    available.append(f)
        raise FileNotFoundError(
            f"SAC model not found at '{model_path}'. "
            f"Available in '{models_dir}': {available if available else 'none'}"
        )

    try:
        model = SAC.load(model_path, env=env)
    except Exception as e:
        raise RuntimeError(
            "Failed to load SAC model. Ensure the file was saved with a\n"
            "compatible stable-baselines3 and gymnasium version.\n"
            f"Model path: {model_path}\nOriginal error: {e}"
        )

    # Aggregators for paper metrics: SR, SPL, CR, FR, CoT
    successes = 0
    spls = []
    collision_eps = 0
    fall_eps = 0
    cots = []
    steps_list = []
    L_actual_list = []
    L_star_list = []
    dist_to_goal_list = []

    rows = []

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        step = 0

        last_info = {}

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            last_info = info if isinstance(info, dict) else {}
            total_reward += float(reward)
            step += 1

        # Get episode metrics from env (preferred)
        m = _safe_get_episode_metrics(env, last_info)

        if m is None:
            print(f"Ep {ep+1:03d}/{episodes} | Steps={step:4d} | Reward={total_reward:.2f} | (no metrics found)")
            continue

        successes += int(m.get("success", False))
        spls.append(float(m.get("spl", 0.0)))
        collision_eps += int(m.get("collision_any", False))
        fall_eps += int(m.get("fall", False))
        steps_list.append(int(m.get("steps", step)))
        L_actual_list.append(float(m.get("L_actual", 0.0)))
        L_star_list.append(float(m.get("L_star", 0.0)))
        dist_to_goal_list.append(float(m.get("dist_to_goal", 0.0)))

        cot_val = float(m.get("cot", float("inf")))
        if math.isfinite(cot_val):
            cots.append(cot_val)

        rows.append({
            "episode": ep + 1,
            "num_obstacles": num_obstacles,
            "success": int(m.get("success", False)),
            "spl": float(m.get("spl", 0.0)),
            "collision_any": int(m.get("collision_any", False)),
            "fall": int(m.get("fall", False)),
            "cot": cot_val,
            "steps": int(m.get("steps", step)),
            "L_actual": float(m.get("L_actual", 0.0)),
            "L_star": float(m.get("L_star", 0.0)),
            "dist_to_goal": float(m.get("dist_to_goal", 0.0)),
            "total_reward": float(total_reward),
        })

        cot_str = "inf" if not math.isfinite(cot_val) else f"{cot_val:.3f}"

        print(
            f"Ep {ep+1:03d}/{episodes} | "
            f"SR={int(m.get('success', False))} | SPL={float(m.get('spl', 0.0)):.3f} | "
            f"CR={int(m.get('collision_any', False))} | FR={int(m.get('fall', False))} | "
            f"CoT={cot_str} | Steps={int(m.get('steps', step)):4d} | Reward={total_reward:.2f}"
        )

    # Summary
    ep_done = len(rows) if rows else max(episodes, 1)
    SR = successes / ep_done
    mean_SPL = float(np.mean(spls)) if spls else 0.0
    CR = collision_eps / ep_done
    FR = fall_eps / ep_done
    mean_CoT = float(np.mean(cots)) if cots else float("inf")
    mean_reward = float(np.mean([r['total_reward'] for r in rows])) if rows else 0.0

    print(f"\n{'='*70}")
    print(f"Evaluation Summary ({num_obstacles} obstacles)")
    print(f"{'='*70}")
    print(f"Episodes evaluated : {ep_done}")
    print(f"Success Rate (SR)  : {SR*100:.2f}%  ({successes}/{ep_done})")
    print(f"Mean SPL           : {mean_SPL:.3f}")
    print(f"Collision Rate (CR): {CR*100:.2f}%  ({collision_eps}/{ep_done})")
    print(f"Fall Rate (FR)     : {FR*100:.2f}%  ({fall_eps}/{ep_done})")
    print(f"Mean CoT           : {'inf' if not math.isfinite(mean_CoT) else f'{mean_CoT:.3f}'}")
    print(f"Mean Reward        : {mean_reward:.2f}")
    if steps_list:
        print(f"Mean Steps         : {float(np.mean(steps_list)):.1f}")
    if L_actual_list:
        print(f"Mean L_actual (m)  : {float(np.mean(L_actual_list)):.3f}")
    if L_star_list:
        print(f"Mean L_star (m)    : {float(np.mean(L_star_list)):.3f}")
    if dist_to_goal_list:
        print(f"Mean dist_to_goal  : {float(np.mean(dist_to_goal_list)):.3f}")
    print(f"{'='*70}\n")

    env.close()

    # Return summary metrics and rows for aggregation
    return {
        "num_obstacles": num_obstacles,
        "SR": SR,
        "mean_SPL": mean_SPL,
        "CR": CR,
        "FR": FR,
        "mean_CoT": mean_CoT,
        "mean_reward": mean_reward,
        "mean_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "mean_L_actual": float(np.mean(L_actual_list)) if L_actual_list else 0.0,
        "mean_L_star": float(np.mean(L_star_list)) if L_star_list else 0.0,
        "mean_dist_to_goal": float(np.mean(dist_to_goal_list)) if dist_to_goal_list else 0.0,
    }, rows


def run_multi_obstacle_tests(
    obstacle_counts=[5, 10, 15, 20],
    model_path: str = DEFAULT_MODEL_PATH,
    episodes: int = 100,
    render: bool = False,
    deterministic: bool = True,
):
    """
    Run tests with multiple obstacle configurations.
    
    Args:
        obstacle_counts: List of obstacle counts to test
        model_path: Path to the trained SAC model
        episodes: Number of episodes per obstacle configuration
        render: Whether to render the environment
        deterministic: Whether to use deterministic actions
    """
    all_summaries = []
    all_rows = []
    
    start_time = time.time()
    
    for num_obs in obstacle_counts:
        summary, rows = test_sac_with_obstacles(
            num_obstacles=num_obs,
            model_path=model_path,
            episodes=episodes,
            render=render,
            deterministic=deterministic,
        )
        all_summaries.append(summary)
        all_rows.extend(rows)
        
        # Small delay between configurations
        time.sleep(1.0)
    
    total_time = time.time() - start_time
    
    # Print overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY - All Obstacle Configurations")
    print(f"{'='*70}")
    print(f"{'Obstacles':<12} {'SR%':<10} {'SPL':<10} {'CR%':<10} {'FR%':<10} {'CoT':<10} {'Reward':<10}")
    print(f"{'-'*70}")
    for s in all_summaries:
        cot_str = "inf" if not math.isfinite(s['mean_CoT']) else f"{s['mean_CoT']:.3f}"
        print(
            f"{s['num_obstacles']:<12} "
            f"{s['SR']*100:<10.2f} "
            f"{s['mean_SPL']:<10.3f} "
            f"{s['CR']*100:<10.2f} "
            f"{s['FR']*100:<10.2f} "
            f"{cot_str:<10} "
            f"{s['mean_reward']:<10.2f}"
        )
    print(f"{'='*70}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"{'='*70}\n")
    
    # Save combined results to CSV
    csv_path = os.path.join(os.path.dirname(__file__), "eval_metrics_multi_obstacles.csv")
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"Saved combined per-episode metrics CSV -> {csv_path}")
    
    # Save summary to separate CSV
    summary_csv_path = os.path.join(os.path.dirname(__file__), "eval_summary_multi_obstacles.csv")
    if all_summaries:
        fieldnames = list(all_summaries[0].keys())
        with open(summary_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for s in all_summaries:
                w.writerow(s)
        print(f"Saved summary metrics CSV -> {summary_csv_path}\n")


if __name__ == "__main__":
    # Run tests with 5, 10, 15, and 20 obstacles
    # Set render=False for faster execution, or render=True to visualize
    run_multi_obstacle_tests(
        obstacle_counts=[5, 10, 15, 20],
        episodes=100,
        render=False,  # Set to True if you want to see the simulation
        deterministic=True,
    )
