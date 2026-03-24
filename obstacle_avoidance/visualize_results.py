#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-quality visualization script for biped walking paper.
Generates multiple figures from test results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Color palette (publication-friendly)
COLORS = {
    '5': '#2E86AB',   # Blue
    '10': '#A23B72',  # Purple
    '15': '#F18F01',  # Orange
    '20': '#C73E1D',  # Red
}


def load_data(csv_path="eval_summary_multi_obstacles.csv"):
    """Load summary data from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_episode_data(csv_path="eval_metrics_multi_obstacles.csv"):
    """Load per-episode data from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


def plot_main_metrics_comparison(df, save_path="fig1_metrics_comparison.png"):
    """
    Figure 1: Main metrics comparison across obstacle configurations.
    4-panel subplot showing SR, SPL, CR, FR.
    """
    # Adjusted figure size for better compactness
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Performance Metrics vs. Obstacle Density', fontsize=14, fontweight='bold', y=0.98)
    
    obstacles = df['num_obstacles'].values
    colors = [COLORS[str(n)] for n in obstacles]
    
    # Success Rate
    ax = axes[0, 0]
    bars = ax.bar(obstacles, df['SR'] * 100, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8, width=3)
    ax.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Number of Obstacles', fontweight='bold', fontsize=11)
    ax.set_title('(a) Success Rate', fontweight='bold', fontsize=12)
    ax.set_ylim([0, 108])  # Increased to prevent text overflow
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xticks(obstacles)
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # SPL
    ax = axes[0, 1]
    bars = ax.bar(obstacles, df['mean_SPL'], color=colors, edgecolor='black', linewidth=1.2, alpha=0.8, width=3)
    ax.set_ylabel('Success-weighted Path Length', fontweight='bold', fontsize=11)
    ax.set_xlabel('Number of Obstacles', fontweight='bold', fontsize=11)
    ax.set_title('(b) Mean SPL', fontweight='bold', fontsize=12)
    ax.set_ylim([0, 1.08])  # Increased to prevent text overflow
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xticks(obstacles)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Collision Rate
    ax = axes[1, 0]
    cr_max = (df['CR'] * 100).max()
    bars = ax.bar(obstacles, df['CR'] * 100, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8, width=3)
    ax.set_ylabel('Collision Rate (%)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Number of Obstacles', fontweight='bold', fontsize=11)
    ax.set_title('(c) Collision Rate', fontweight='bold', fontsize=12)
    ax.set_ylim([0, cr_max * 1.35])  # Dynamic limit to prevent overflow
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xticks(obstacles)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + cr_max * 0.05,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Fall Rate
    ax = axes[1, 1]
    fr_max = (df['FR'] * 100).max()
    bars = ax.bar(obstacles, df['FR'] * 100, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8, width=3)
    ax.set_ylabel('Fall Rate (%)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Number of Obstacles', fontweight='bold', fontsize=11)
    ax.set_title('(d) Fall Rate', fontweight='bold', fontsize=12)
    ax.set_ylim([0, fr_max * 1.35])  # Dynamic limit to prevent overflow
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xticks(obstacles)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + fr_max * 0.05,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust to accommodate suptitle
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_cost_of_transport(df, save_path="fig2_cost_of_transport.png"):
    """
    Figure 2: Cost of Transport analysis.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    obstacles = df['num_obstacles'].values
    cot = df['mean_CoT'].values
    colors = [COLORS[str(n)] for n in obstacles]
    
    bars = ax.bar(obstacles, cot, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8, width=3)
    ax.set_ylabel('Cost of Transport (CoT)', fontweight='bold', fontsize=13)
    ax.set_xlabel('Number of Obstacles', fontweight='bold', fontsize=13)
    ax.set_title('Energy Efficiency vs. Obstacle Density', fontweight='bold', fontsize=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xticks(obstacles)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add reference line for baseline
    if len(cot) > 0:
        baseline = cot[0]
        ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Baseline (5 obs)')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_path_efficiency(df, save_path="fig3_path_efficiency.png"):
    """
    Figure 3: Path efficiency analysis (L_actual vs L_star).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    obstacles = df['num_obstacles'].values
    x = np.arange(len(obstacles))
    width = 0.35
    
    colors_actual = [COLORS[str(n)] for n in obstacles]
    colors_star = ['#95a5a6' for _ in obstacles]  # Gray for optimal
    
    bars1 = ax.bar(x - width/2, df['mean_L_actual'], width, label='Actual Path Length',
                   color=colors_actual, edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, df['mean_L_star'], width, label='Optimal Path Length',
                   color=colors_star, edgecolor='black', linewidth=1.5, alpha=0.8, hatch='//')
    
    ax.set_ylabel('Path Length (meters)', fontweight='bold', fontsize=13)
    ax.set_xlabel('Number of Obstacles', fontweight='bold', fontsize=13)
    ax.set_title('Actual vs. Optimal Path Length', fontweight='bold', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(obstacles)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_episode_distribution(df_episodes, save_path="fig4_episode_distribution.png"):
    """
    Figure 4: Distribution of success across episodes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Episode-wise Performance Distribution', fontsize=16, fontweight='bold')
    
    obstacle_counts = sorted(df_episodes['num_obstacles'].unique())
    
    for idx, n_obs in enumerate(obstacle_counts):
        ax = axes[idx // 2, idx % 2]
        data = df_episodes[df_episodes['num_obstacles'] == n_obs]
        
        # Success over episodes
        success = data['success'].values
        episodes = data['episode'].values
        
        colors = [COLORS[str(n_obs)] if s == 1 else '#e74c3c' for s in success]
        ax.scatter(episodes, success, c=colors, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add trend line
        success_rate_cumulative = np.cumsum(success) / np.arange(1, len(success) + 1)
        ax.plot(episodes, success_rate_cumulative, color=COLORS[str(n_obs)], 
                linewidth=2.5, label=f'Cumulative SR', alpha=0.8)
        
        ax.set_ylabel('Success / Cumulative SR', fontweight='bold')
        ax.set_xlabel('Episode Number', fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {n_obs} Obstacles', fontweight='bold')
        ax.set_ylim([-0.1, 1.1])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend()
        
        # Add final SR text
        final_sr = success.mean() * 100
        ax.text(0.98, 0.02, f'Final SR: {final_sr:.1f}%', 
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_spl_boxplot(df_episodes, save_path="fig5_spl_boxplot.png"):
    """
    Figure 5: SPL distribution across obstacle configurations (boxplot).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    obstacle_counts = sorted(df_episodes['num_obstacles'].unique())
    data_to_plot = []
    positions = []
    colors_list = []
    
    for n_obs in obstacle_counts:
        subset = df_episodes[df_episodes['num_obstacles'] == n_obs]
        # Only include successful episodes for SPL
        successful = subset[subset['success'] == 1]
        data_to_plot.append(successful['spl'].values)
        positions.append(n_obs)
        colors_list.append(COLORS[str(n_obs)])
    
    bp = ax.boxplot(data_to_plot, positions=positions, widths=2.5,
                    patch_artist=True, showfliers=True,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='darkred'))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
    
    ax.set_ylabel('Success-weighted Path Length (SPL)', fontweight='bold', fontsize=13)
    ax.set_xlabel('Number of Obstacles', fontweight='bold', fontsize=13)
    ax.set_title('SPL Distribution Across Obstacle Densities', fontweight='bold', fontsize=15)
    ax.set_xticks(positions)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_comprehensive_table(df, save_path="fig6_results_table.png"):
    """
    Figure 6: Comprehensive results table with proper units and academic styling.
    """
    # Create the table data
    table_data = []
    # Updated headers with explicit units
    headers = [
        'Obstacles', 'SR (%)', 'Mean SPL', 'CR (%)', 'FR (%)', 
        'CoT', 'Avg Reward', 'Path Dev (m)', 'L_act (m)', 'L_opt (m)', 'Dist (m)'
    ]
    
    for _, row in df.iterrows():
        table_data.append([
            int(row['num_obstacles']),
            f"{row['SR']*100:.1f}",
            f"{row['mean_SPL']:.3f}",
            f"{row['CR']*100:.1f}",
            f"{row['FR']*100:.1f}",
            f"{row['mean_CoT']:.2f}",
            f"{row['mean_reward']:.1f}",
            f"{row['mean_path_dev']:.2f} ± {row['std_path_dev']:.2f}",
            f"{row['mean_L_actual']:.2f}",
            f"{row['mean_L_star']:.2f}",
            f"{row['mean_dist_to_goal']:.3f}"
        ])
    
    # Using a larger figure and tighter scaling to prevent text overflow
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis('off')
    
    # Create the table
    the_table = ax.table(cellText=table_data,
                         colLabels=headers,
                         loc='center',
                         cellLoc='center',
                         colColours=['#34495e']*len(headers))
    
    # Style the table
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1.0, 2.2)
    
    # Color headers
    for i in range(len(headers)):
        cell = the_table[(0, i)]
        cell.get_text().set_color('white')
        cell.get_text().set_weight('bold')
    
    # Color rows based on obstacle count for consistency
    for i, row_data in enumerate(table_data, start=1):
        n_obs = row_data[0]
        color = COLORS[str(n_obs)]
        for j in range(len(headers)):
            cell = the_table[(i, j)]
            cell.set_facecolor(color)
            cell.set_alpha(0.15)
            cell.set_edgecolor('#bdc3c7')
    
    plt.title('Summary of Evaluation Metrics Across Obstacle Configurations',
              fontweight='bold', fontsize=13, pad=15)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_performance_trends(df, save_path="fig7_performance_trends.png"):
    """
    Figure 7: Performance trends as obstacle density increases.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    obstacles = df['num_obstacles'].values
    
    # Normalize metrics to 0-100 scale for comparison
    sr_norm = df['SR'] * 100
    spl_norm = df['mean_SPL'] * 100
    cot_norm = 100 - (df['mean_CoT'] - df['mean_CoT'].min()) / (df['mean_CoT'].max() - df['mean_CoT'].min() + 1e-6) * 50
    
    ax.plot(obstacles, sr_norm, marker='o', linewidth=2.5, markersize=10, 
            label='Success Rate', color='#2E86AB', linestyle='-')
    ax.plot(obstacles, spl_norm, marker='s', linewidth=2.5, markersize=10,
            label='SPL (×100)', color='#A23B72', linestyle='-')
    ax.plot(obstacles, cot_norm, marker='^', linewidth=2.5, markersize=10,
            label='CoT Efficiency', color='#F18F01', linestyle='-')
    
    ax.set_ylabel('Performance Score', fontweight='bold', fontsize=13)
    ax.set_xlabel('Number of Obstacles', fontweight='bold', fontsize=13)
    ax.set_title('Performance Trends with Increasing Obstacle Density', fontweight='bold', fontsize=15)
    ax.set_xticks(obstacles)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9, fontsize=11)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def generate_all_figures():
    """Generate all publication figures."""
    print("\n" + "="*70)
    print("Generating Publication-Quality Figures")
    print("="*70 + "\n")
    
    # Check if summary data exists
    summary_path = "eval_summary_multi_obstacles.csv"
    episode_path = "eval_metrics_multi_obstacles.csv"
    
    if not os.path.exists(summary_path):
        print(f"❌ Error: {summary_path} not found!")
        print("Please run the multi-obstacle test first.")
        return
    
    # Load data
    print("Loading data...")
    df_summary = load_data(summary_path)
    
    # Generate figures
    print("\nGenerating figures...\n")
    
    plot_main_metrics_comparison(df_summary)
    plot_cost_of_transport(df_summary)
    plot_path_efficiency(df_summary)
    plot_performance_trends(df_summary)
    plot_comprehensive_table(df_summary)
    
    if os.path.exists(episode_path):
        df_episodes = load_episode_data(episode_path)
        plot_episode_distribution(df_episodes)
        plot_spl_boxplot(df_episodes)
    else:
        print(f"⚠ Warning: {episode_path} not found. Skipping episode-level plots.")
    
    print("\n" + "="*70)
    print("✓ All figures generated successfully!")
    print("="*70 + "\n")
    print("Generated figures:")
    print("  • fig1_metrics_comparison.png - Main metrics (SR, SPL, CR, FR)")
    print("  • fig2_cost_of_transport.png - Energy efficiency analysis")
    print("  • fig3_path_efficiency.png - Path length comparison")
    print("  • fig4_episode_distribution.png - Episode-wise performance")
    print("  • fig5_spl_boxplot.png - SPL distribution")
    print("  • fig6_results_table.png - Summary table")
    print("  • fig7_performance_trends.png - Performance trends")
    print("\nAll figures are saved at 300 DPI for publication quality.")
    print("="*70 + "\n")


if __name__ == "__main__":
    generate_all_figures()
