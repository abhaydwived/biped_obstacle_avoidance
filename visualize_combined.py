#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined figures for academic papers - saves space and shows related metrics together.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Publication-quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Color palette
COLORS = {
    '5': '#2E86AB',
    '10': '#A23B72',
    '15': '#F18F01',
    '20': '#C73E1D',
}


def load_data(summary_path="eval_summary_multi_obstacles.csv", 
              episode_path="eval_metrics_multi_obstacles.csv"):
    """Load both summary and episode data."""
    df_summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else None
    df_episodes = pd.read_csv(episode_path) if os.path.exists(episode_path) else None
    return df_summary, df_episodes


def plot_combined_efficiency_analysis(df, save_path="fig_combined_efficiency.png"):
    """
    Combined Figure: Energy Efficiency & Path Planning
    (a) Cost of Transport
    (b) Path Efficiency (Actual vs Optimal)
    (c) Performance Trends
    """
    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    obstacles = df['num_obstacles'].values
    colors_list = [COLORS[str(n)] for n in obstacles]
    
    # (a) Cost of Transport
    ax1 = fig.add_subplot(gs[0, 0])
    cot = df['mean_CoT'].values
    bars = ax1.bar(obstacles, cot, color=colors_list, edgecolor='black', 
                   linewidth=1.2, alpha=0.8, width=3)
    ax1.set_ylabel('Cost of Transport (CoT)', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Number of Obstacles', fontweight='bold', fontsize=11)
    ax1.set_title('(a) Energy Efficiency', fontweight='bold', fontsize=12)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_xticks(obstacles)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Baseline reference
    baseline = cot[0]
    ax1.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.2, 
                alpha=0.6, label=f'Baseline ({obstacles[0]} obs)')
    ax1.legend(loc='upper left', fontsize=8)
    
    # (b) Path Efficiency
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(obstacles))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, df['mean_L_actual'], width, label='Actual Path',
                   color=colors_list, edgecolor='black', linewidth=1.2, alpha=0.8)
    bars2 = ax2.bar(x + width/2, df['mean_L_star'], width, label='Optimal Path',
                   color='#95a5a6', edgecolor='black', linewidth=1.2, alpha=0.8, hatch='//')
    
    ax2.set_ylabel('Path Length (meters)', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Number of Obstacles', fontweight='bold', fontsize=11)
    ax2.set_title('(b) Path Planning Efficiency', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(obstacles)
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # (c) Performance Trends
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Normalize metrics for comparison
    sr_norm = df['SR'] * 100
    spl_norm = df['mean_SPL'] * 100
    
    ax3.plot(obstacles, sr_norm, marker='o', linewidth=2.5, markersize=8, 
            label='Success Rate (%)', color='#2E86AB', linestyle='-')
    ax3.plot(obstacles, spl_norm, marker='s', linewidth=2.5, markersize=8,
            label='SPL (×100)', color='#A23B72', linestyle='-')
    
    ax3.set_ylabel('Performance Score', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Number of Obstacles', fontweight='bold', fontsize=11)
    ax3.set_title('(c) Performance Degradation', fontweight='bold', fontsize=12)
    ax3.set_xticks(obstacles)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='lower left', framealpha=0.9, fontsize=9)
    ax3.set_ylim([85, 105])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_combined_robustness_analysis(df_summary, df_episodes, 
                                       save_path="fig_combined_robustness.png"):
    """
    Combined Figure: Robustness Analysis
    (a-d) Episode success patterns (4 obstacle configurations)
    (e) SPL distribution boxplot
    (f) Success rate summary
    """
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    obstacle_counts = sorted(df_episodes['num_obstacles'].unique())
    
    # (a-d) Episode distributions - 4 subplots
    for idx, n_obs in enumerate(obstacle_counts):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        data = df_episodes[df_episodes['num_obstacles'] == n_obs]
        success = data['success'].values
        episodes = data['episode'].values
        
        colors = [COLORS[str(n_obs)] if s == 1 else '#e74c3c' for s in success]
        ax.scatter(episodes, success, c=colors, s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
        
        # Cumulative success rate
        success_rate_cumulative = np.cumsum(success) / np.arange(1, len(success) + 1)
        ax.plot(episodes, success_rate_cumulative, color=COLORS[str(n_obs)], 
                linewidth=2, label=f'Cumulative SR', alpha=0.8)
        
        ax.set_ylabel('Success / Cumulative SR', fontweight='bold', fontsize=10)
        ax.set_xlabel('Episode', fontweight='bold', fontsize=10)
        ax.set_title(f'({chr(97+idx)}) {n_obs} Obstacles', fontweight='bold', fontsize=11)
        ax.set_ylim([-0.1, 1.1])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=8)
        
        # Final SR text
        final_sr = success.mean() * 100
        ax.text(0.98, 0.02, f'SR: {final_sr:.1f}%', 
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontweight='bold', fontsize=8)
    
    # (e) SPL Boxplot - bottom left
    ax_box = fig.add_subplot(gs[1, 1])
    data_to_plot = []
    positions = []
    colors_list = []
    
    for n_obs in obstacle_counts:
        subset = df_episodes[df_episodes['num_obstacles'] == n_obs]
        successful = subset[subset['success'] == 1]
        data_to_plot.append(successful['spl'].values)
        positions.append(n_obs)
        colors_list.append(COLORS[str(n_obs)])
    
    bp = ax_box.boxplot(data_to_plot, positions=positions, widths=2.5,
                        patch_artist=True, showfliers=True,
                        boxprops=dict(linewidth=1.2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        medianprops=dict(linewidth=2, color='darkred'))
    
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
    
    ax_box.set_ylabel('SPL', fontweight='bold', fontsize=11)
    ax_box.set_xlabel('Number of Obstacles', fontweight='bold', fontsize=11)
    ax_box.set_title('(e) SPL Distribution', fontweight='bold', fontsize=12)
    ax_box.set_xticks(positions)
    ax_box.grid(axis='y', alpha=0.3, linestyle='--')
    ax_box.set_ylim([0.8, 1.02])
    
    # (f) Summary metrics - bottom right
    ax_summary = fig.add_subplot(gs[1, 2])
    x = np.arange(len(obstacle_counts))
    width = 0.25
    
    obstacles = df_summary['num_obstacles'].values
    sr_vals = df_summary['SR'].values * 100
    cr_vals = df_summary['CR'].values * 100
    fr_vals = df_summary['FR'].values * 100
    
    bars1 = ax_summary.bar(x - width, sr_vals, width, label='Success Rate',
                          color='#27ae60', edgecolor='black', linewidth=1, alpha=0.8)
    bars2 = ax_summary.bar(x, cr_vals, width, label='Collision Rate',
                          color='#e67e22', edgecolor='black', linewidth=1, alpha=0.8)
    bars3 = ax_summary.bar(x + width, fr_vals, width, label='Fall Rate',
                          color='#e74c3c', edgecolor='black', linewidth=1, alpha=0.8)
    
    ax_summary.set_ylabel('Rate (%)', fontweight='bold', fontsize=11)
    ax_summary.set_xlabel('Number of Obstacles', fontweight='bold', fontsize=11)
    ax_summary.set_title('(f) Success vs. Failure Rates', fontweight='bold', fontsize=12)
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(obstacles)
    ax_summary.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax_summary.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('Robustness Analysis Across Obstacle Densities', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_combined_compact(df_summary, df_episodes, save_path="fig_combined_compact.png"):
    """
    Most compact combined figure for papers with strict page limits.
    2×2 layout with key metrics only.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Comprehensive Performance Analysis', fontsize=14, fontweight='bold', y=0.98)
    
    obstacles = df_summary['num_obstacles'].values
    colors_list = [COLORS[str(n)] for n in obstacles]
    
    # (a) SR + SPL combined
    ax = axes[0, 0]
    ax2 = ax.twinx()
    
    bars = ax.bar(obstacles - 1, df_summary['SR'] * 100, width=1.8, 
                  label='Success Rate', color='#2E86AB', edgecolor='black', 
                  linewidth=1.2, alpha=0.8)
    line = ax2.plot(obstacles, df_summary['mean_SPL'], marker='s', linewidth=2.5, 
                    markersize=8, label='SPL', color='#A23B72', linestyle='-')
    
    ax.set_ylabel('Success Rate (%)', fontweight='bold', color='#2E86AB', fontsize=11)
    ax2.set_ylabel('Mean SPL', fontweight='bold', color='#A23B72', fontsize=11)
    ax.set_xlabel('Number of Obstacles', fontweight='bold', fontsize=11)
    ax.set_title('(a) Success Metrics', fontweight='bold', fontsize=12)
    ax.set_xticks(obstacles)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([85, 105])
    ax2.set_ylim([0.85, 1.0])
    
    # Legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=9)
    
    # (b) CoT + Path Efficiency
    ax = axes[0, 1]
    ax2 = ax.twinx()
    
    bars = ax.bar(obstacles, df_summary['mean_CoT'], color=colors_list, 
                  edgecolor='black', linewidth=1.2, alpha=0.8, width=3)
    
    efficiency = (df_summary['mean_L_star'] / df_summary['mean_L_actual']) * 100
    line = ax2.plot(obstacles, efficiency, marker='o', linewidth=2.5, markersize=8,
                    color='#27ae60', linestyle='-', label='Path Efficiency')
    
    ax.set_ylabel('Cost of Transport', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Path Efficiency (%)', fontweight='bold', color='#27ae60', fontsize=11)
    ax.set_xlabel('Number of Obstacles', fontweight='bold', fontsize=11)
    ax.set_title('(b) Efficiency Metrics', fontweight='bold', fontsize=12)
    ax.set_xticks(obstacles)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(loc='lower left', fontsize=9)
    
    # (c) Failure rates
    ax = axes[1, 0]
    x = np.arange(len(obstacles))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df_summary['CR'] * 100, width, label='Collision Rate',
                   color='#e67e22', edgecolor='black', linewidth=1.2, alpha=0.8)
    bars2 = ax.bar(x + width/2, df_summary['FR'] * 100, width, label='Fall Rate',
                   color='#e74c3c', edgecolor='black', linewidth=1.2, alpha=0.8)
    
    ax.set_ylabel('Failure Rate (%)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Number of Obstacles', fontweight='bold', fontsize=11)
    ax.set_title('(c) Failure Analysis', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(obstacles)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # (d) SPL boxplot
    ax = axes[1, 1]
    data_to_plot = []
    positions = []
    
    for n_obs in obstacles:
        subset = df_episodes[df_episodes['num_obstacles'] == n_obs]
        successful = subset[subset['success'] == 1]
        data_to_plot.append(successful['spl'].values)
        positions.append(n_obs)
    
    bp = ax.boxplot(data_to_plot, positions=positions, widths=2.5,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(linewidth=1.2),
                    whiskerprops=dict(linewidth=1.2),
                    medianprops=dict(linewidth=2, color='darkred'))
    
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
    
    ax.set_ylabel('SPL', fontweight='bold', fontsize=11)
    ax.set_xlabel('Number of Obstacles', fontweight='bold', fontsize=11)
    ax.set_title('(d) SPL Distribution', fontweight='bold', fontsize=12)
    ax.set_xticks(positions)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"✓ Saved: {save_path}")
    plt.close()


def generate_combined_figures():
    """Generate all combined figures."""
    print("\n" + "="*70)
    print("Generating Combined Academic Figures")
    print("="*70 + "\n")
    
    # Load data
    df_summary, df_episodes = load_data()
    
    if df_summary is None:
        print("❌ Error: eval_summary_multi_obstacles.csv not found!")
        return
    
    print("Generating combined figures...\n")
    
    # Generate combined figures
    plot_combined_efficiency_analysis(df_summary)
    
    if df_episodes is not None:
        plot_combined_robustness_analysis(df_summary, df_episodes)
        plot_combined_compact(df_summary, df_episodes)
    
    print("\n" + "="*70)
    print("✓ All combined figures generated!")
    print("="*70 + "\n")
    print("Generated files:")
    print("  • fig_combined_efficiency.png - Energy & Path analysis (3-panel)")
    print("  • fig_combined_robustness.png - Robustness analysis (6-panel)")
    print("  • fig_combined_compact.png - Compact version (2×2 grid)")
    print("\nAll figures saved at 300 DPI for publication quality.")
    print("="*70 + "\n")


if __name__ == "__main__":
    generate_combined_figures()
