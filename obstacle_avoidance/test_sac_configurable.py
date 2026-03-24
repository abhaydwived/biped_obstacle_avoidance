#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to test SAC with a configurable number of obstacles.
Just change NUM_OBSTACLES below to 5, 10, 15, or 20 and run.
"""

import os
import time
import math
import csv
import numpy as np
from stable_baselines3 import SAC
from biped_env_obs3 import BipedWalkingEnv


# ============================================================================
# CONFIGURATION - Change these values as needed
# ============================================================================
NUM_OBSTACLES = 16  # Change this to 5, 10, 15, or 20
EPISODES = 100
RENDER = False  # Set to True to see the simulation GUI
DETERMINISTIC = True
# ============================================================================


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


def test_sac_eval(
    model_path: str = DEFAULT_MODEL_PATH,
    episodes: int = EPISODES,
    render: bool = RENDER,
    deterministic: bool = DETERMINISTIC,
    save_csv: bool = True,
    keep_window_open: bool = False,
    num_obstacles: int = NUM_OBSTACLES,
):
    env = BipedWalkingEnv(render=render)
    
    # Override the number of obstacles
    env.num_obstacles = num_obstacles
    
    print(f"\n{'='*70}")
    print(f"[Eval] Testing with {num_obstacles} obstacles")
    print(f"[Eval] PyBullet render={env.render} (render=True => GUI window should appear)")
    print(f"{'='*70}\n")

    # Helpful diagnostics if model is missing
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

    print(f"\n{'='*70}")
    print(f"Evaluation Summary ({num_obstacles} obstacles)")
    print(f"{'='*70}")
    print(f"Episodes evaluated : {ep_done}")
    print(f"Success Rate (SR)  : {SR*100:.2f}%  ({successes}/{ep_done})")
    print(f"Mean SPL           : {mean_SPL:.3f}")
    print(f"Collision Rate (CR): {CR*100:.2f}%  ({collision_eps}/{ep_done})")
    print(f"Fall Rate (FR)     : {FR*100:.2f}%  ({fall_eps}/{ep_done})")
    print(f"Mean CoT           : {'inf' if not math.isfinite(mean_CoT) else f'{mean_CoT:.3f}'}")
    if steps_list:
        print(f"Mean Steps         : {float(np.mean(steps_list)):.1f}")
    if L_actual_list:
        print(f"Mean L_actual (m)  : {float(np.mean(L_actual_list)):.3f}")
    if L_star_list:
        print(f"Mean L_star (m)    : {float(np.mean(L_star_list)):.3f}")
    if dist_to_goal_list:
        print(f"Mean dist_to_goal  : {float(np.mean(dist_to_goal_list)):.3f}")
    print(f"{'='*70}\n")

    # Save CSV for paper plots/tables
    if save_csv and rows:
        csv_path = os.path.join(os.path.dirname(__file__), f"eval_metrics_{num_obstacles}obs.csv")
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Saved per-episode metrics CSV -> {csv_path}")

    # Keep PyBullet GUI open so you can see the final scene
    if render and keep_window_open:
        try:
            input("PyBullet GUI is open. Press Enter to close and exit...")
        except EOFError:
            # In some environments stdin may not be available; just sleep a bit.
            time.sleep(2.0)

    env.close()


if __name__ == "__main__":
    # Run evaluation with the number of obstacles specified at the top
    test_sac_eval()
