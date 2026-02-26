#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import csv
import numpy as np
from stable_baselines3 import SAC

# IMPORTANT:
# If you saved the updated env as biped_env_obs3_metrics.py, use this import:
from biped_env_obs2 import BipedWalkingEnv
# If your file name is still biped_env_obs3.py but you pasted the metric code inside it,
# then change the import line above to:
# from biped_env_obs3 import BipedWalkingEnv


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


def test_sac_eval(
    model_path: str = DEFAULT_MODEL_PATH,
    episodes: int = 100,
    render: bool = True,
    deterministic: bool = True,
    save_csv: bool = True,
    keep_window_open: bool = True,
):
    env = BipedWalkingEnv(render=render)
    print(f"[Eval] PyBullet render={env.render} (render=True => GUI window should appear)")

    # Helpful diagnostics if model is missing
    if not os.path.isfile(model_path):
        available = []
        models_dir = os.path.join(os.path.dirname(__file__), "models_obs2")
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

    print("\n==================== Evaluation Summary ====================")
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
    print("===========================================================\n")

    # Save CSV for paper plots/tables
    if save_csv and rows:
        csv_path = os.path.join(os.path.dirname(__file__), "eval_metrics_100eps.csv")
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
    # GUI ON by default for visibility
    test_sac_eval(episodes=100, render=True, deterministic=True, save_csv=True, keep_window_open=True)
