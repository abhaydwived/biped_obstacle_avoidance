# Underactuated Biped Robot Obstacle Avoidance using Deep Reinforcement Learning

> **Note:** This repository contains the official codebase and results for the paper currently under review at **IJCAI-ECAI 2026**.

## 🎥 Simulation Demonstration

<div align="center">
  <video src="Experiment%20videos/exp_final_pass.mp4" width="80%" controls autoplay loop muted></video>
</div>

## 📌 Overview

This project presents a robust and energy-efficient obstacle avoidance framework for an 8-DOF bipedal robot using Deep Reinforcement Learning (Soft Actor-Critic). By tightly integrating an A* planner with a responsive local control policy, the framework successfully navigates through densely cluttered environments while maintaining stability, minimizing energy consumption (Cost of Transport), and avoiding collisions or falls.

The framework is tested across various obstacle densities (5, 10, 15, and 20 obstacles) within a custom PyBullet simulation environment that provides a rigorous evaluation suite for standard metrics.

## ✨ Key Features

- **Custom PyBullet Environment**: High-fidelity bipedal locomotion simulation built on `Gymnasium` with realistic physics and contact models.
- **Hierarchical Navigation Scheme**: Fuses global A* path planning with a local, reactive Soft Actor-Critic (SAC) control policy augmented by Artificial Potential Fields (APF).
- **Comprehensive Metric Tracking**: Automatically computes Success Rate (SR), Success weighted by Path Length (SPL), Cost of Transport (CoT), Collision Rate (CR), Fall Rate (FR), and efficiency metrics directly comparable to established baselines.
- **Automated Figure Generation**: Ready-to-use scripts to generate high-quality, combined LaTeX-ready figures (`visualize_combined.py` and `COMBINED_FIGURES_GUIDE.md`) suited for academic publishing with strict page limits.

## 📁 Repository Structure

```
├── obstacle_avoidance/
│   ├── biped_env_obs2.py / biped_env_obs3.py # Custom Gymnasium Environments (PyBullet)
│   ├── train_sac.py                          # Training script using Stable-Baselines3
│   ├── test_sac_multi_obstacles.py           # Evaluation script across different obstacle densities
│   ├── callbacks.py                          # Logging and reward plotting callbacks
│   ├── visualize_results.py                  # Single-figure generation script
│   ├── visualize_combined.py                 # Multi-panel combined figure generation
│   ├── COMBINED_FIGURES_GUIDE.md             # Guide on using the visualization scripts
│   ├── models_obs2/ / models_obs3/           # Trained SAC policy models
│   ├── logs/                                 # TensorBoard and monitor logs
│   ├── bi_urdf/ / yaw_up/                    # Robot URDF files & assets
│   └── run_multi_obstacle_tests.bat          # Batch script for automated evaluation
├── all_tables.tex                            # LaTeX code for all academic tables
└── README.md                                 # This documentation file
```

## 🛠️ Installation

**Prerequisites:** Python 3.8+

Clone this repository and install the necessary dependencies:

```bash
git clone <anonymous-repo-url>
cd biped_obstacle_avoidance
pip install -r requirements.txt
```

*(Note: Ensure you have installed packages like `gymnasium`, `pybullet`, `stable-baselines3`, `numpy`, and `matplotlib`.)*

## 🚀 Usage

### 1. Training the Policy

To train the Soft Actor-Critic (SAC) policy from scratch:
```bash
cd obstacle_avoidance
python train_sac.py
```
*Logs and checkpoints will be saved in the `logs/` and `models_obs3/` directories.*

### 2. Evaluating the Policy

To evaluate the trained model across customized obstacle densities and record the standard metrics:
```bash
cd obstacle_avoidance
python test_sac_multi_obstacles.py
```

### 3. Generating Results and Figures

For the **IJCAI-ECAI 2026** paper, we combined multiple evaluation metrics into space-efficient figures. To generate these comprehensive sets of figures:
```bash
cd obstacle_avoidance
python visualize_combined.py
```
Please refer to `obstacle_avoidance/COMBINED_FIGURES_GUIDE.md` for specific instructions on integrating these generated figures (`fig_combined_efficiency.png`, `fig_combined_robustness.png`, `fig_combined_compact.png`) into LaTeX source limits.

## 📊 Results Summary

Performance metrics across different obstacle densities. All values represent means over 100 evaluation episodes.

| Obstacles | SR (%) | SPL   | CR (%) | FR (%) | CoT   | Path Dev (m) | Reward |
|-----------|--------|-------|--------|--------|-------|--------------|--------|
| **5**     | 100.0  | 0.975 | 1.0    | 0.0    | 14.40 | 0.15 ± 0.09  | 2487.2 |
| **10**    | 98.0   | 0.955 | 3.0    | 2.0    | 14.33 | 0.09 ± 0.58  | 2380.2 |
| **15**    | 96.0   | 0.937 | 5.0    | 4.0    | 14.42 | 0.02 ± 0.71  | 2327.0 |
| **20**    | 94.0   | 0.919 | 5.0    | 6.0    | 14.22 | -0.10 ± 1.10 | 2286.0 |

*Please see `all_tables.tex` for the complete sets of statistical analysis and baseline comparison tables referenced in the paper.*

## 📜 License & Citation

*(Repository is currently under double-blind peer review for IJCAI-ECAI 2026. Code will be released under an open-source license upon acceptance).*
