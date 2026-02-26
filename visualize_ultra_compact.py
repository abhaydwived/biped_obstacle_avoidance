#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-compact academic figures - Maximum information, minimum space.
Perfect for conference papers with strict page limits.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Publication-quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8

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


def plot_ultra_compact_single(df_summary, df_episodes, 
                               save_path="fig_paper_main.png"):
    """
    SINGLE ultra-compact figure with ALL key metrics.
    3×2 grid showing everything needed for results section.
    """
    fig = plt.figure(figsize=(11, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    obstacles = df_summary['num_obstacles'].values
    colors_list = [COLORS[str(n)] for n in obstacles]
    
    # ==================== ROW 1: Main Performance Metrics ====================
    
    # (a) Success Rate + SPL (dual y-axis)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_twin = ax1.twinx()
    
    # Success Rate bars
    bars = ax1.bar(obstacles, df_summary['SR'] * 100, color=colors_list, 
                   edgecolor='black', linewidth=1, alpha=0.75, width=3, 
                   label='Success Rate', zorder=2)
    
    # SPL line
    line = ax1_twin.plot(obstacles, df_summary['mean_SPL'], marker='s', 
                         linewidth=2, markersize=6, label='SPL', 
                         color='darkgreen', linestyle='-', zorder=3)
    
    ax1.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=9, color='black')
    ax1_twin.set_ylabel('SPL', fontweight='bold', fontsize=9, color='darkgreen')
    ax1.set_xlabel('Obstacles', fontweight='bold', fontsize=9)
    ax1.set_title('(a) Success Metrics', fontweight='bold', fontsize=10)
    ax1.set_xticks(obstacles)
    ax1.grid(axis='y', alpha=0.25, linestyle='--', zorder=1)
    ax1.set_ylim([88, 102])
    ax1_twin.set_ylim([0.88, 1.0])
    ax1_twin.tick_params(axis='y', labelcolor='darkgreen')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=7)
    
    # (b) Collision & Fall Rates
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(obstacles))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, df_summary['CR'] * 100, width, 
                    label='Collision', color='#e67e22', edgecolor='black', 
                    linewidth=1, alpha=0.75)
    bars2 = ax2.bar(x + width/2, df_summary['FR'] * 100, width, 
                    label='Fall', color='#e74c3c', edgecolor='black', 
                    linewidth=1, alpha=0.75)
    
    ax2.set_ylabel('Failure Rate (%)', fontweight='bold', fontsize=9)
    ax2.set_xlabel('Obstacles', fontweight='bold', fontsize=9)
    ax2.set_title('(b) Failure Modes', fontweight='bold', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(obstacles)
    ax2.legend(fontsize=7, framealpha=0.9, loc='upper left')
    ax2.grid(axis='y', alpha=0.25, linestyle='--')
    
    # (c) Energy & Path Efficiency (dual y-axis)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3_twin = ax3.twinx()
    
    # CoT bars
    bars = ax3.bar(obstacles, df_summary['mean_CoT'], color=colors_list, 
                   edgecolor='black', linewidth=1, alpha=0.75, width=3,
                   label='CoT', zorder=2)
    
    # Path efficiency line
    efficiency = (df_summary['mean_L_star'] / df_summary['mean_L_actual']) * 100
    line = ax3_twin.plot(obstacles, efficiency, marker='o', linewidth=2, 
                         markersize=6, color='#27ae60', linestyle='-', 
                         label='Path Eff.', zorder=3)
    
    ax3.set_ylabel('CoT', fontweight='bold', fontsize=9)
    ax3_twin.set_ylabel('Path Efficiency (%)', fontweight='bold', fontsize=9, 
                        color='#27ae60')
    ax3.set_xlabel('Obstacles', fontweight='bold', fontsize=9)
    ax3.set_title('(c) Efficiency Metrics', fontweight='bold', fontsize=10)
    ax3.set_xticks(obstacles)
    ax3.grid(axis='y', alpha=0.25, linestyle='--', zorder=1)
    ax3_twin.tick_params(axis='y', labelcolor='#27ae60')
    ax3_twin.set_ylim([94, 101])
    
    # Combined legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=7)
    
    # ==================== ROW 2: Detailed Analysis ====================
    
    # (d) SPL Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    if df_episodes is not None:
        data_to_plot = []
        positions = []
        
        for n_obs in obstacles:
            subset = df_episodes[df_episodes['num_obstacles'] == n_obs]
            successful = subset[subset['success'] == 1]
            data_to_plot.append(successful['spl'].values)
            positions.append(n_obs)
        
        bp = ax4.boxplot(data_to_plot, positions=positions, widths=2.5,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(linewidth=1),
                        whiskerprops=dict(linewidth=1),
                        medianprops=dict(linewidth=1.5, color='darkred'))
        
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor('black')
    
    ax4.set_ylabel('SPL', fontweight='bold', fontsize=9)
    ax4.set_xlabel('Obstacles', fontweight='bold', fontsize=9)
    ax4.set_title('(d) SPL Distribution', fontweight='bold', fontsize=10)
    ax4.set_xticks(obstacles)
    ax4.grid(axis='y', alpha=0.25, linestyle='--')
    ax4.set_ylim([0.85, 1.02])
    
    # (e) Performance Trends
    ax5 = fig.add_subplot(gs[1, 1])
    
    sr_norm = df_summary['SR'] * 100
    spl_norm = df_summary['mean_SPL'] * 100
    
    ax5.plot(obstacles, sr_norm, marker='o', linewidth=2, markersize=6, 
            label='SR (%)', color='#2E86AB', linestyle='-')
    ax5.plot(obstacles, spl_norm, marker='s', linewidth=2, markersize=6,
            label='SPL (×100)', color='#A23B72', linestyle='--')
    
    ax5.set_ylabel('Performance Score', fontweight='bold', fontsize=9)
    ax5.set_xlabel('Obstacles', fontweight='bold', fontsize=9)
    ax5.set_title('(e) Performance Trends', fontweight='bold', fontsize=10)
    ax5.set_xticks(obstacles)
    ax5.grid(True, alpha=0.25, linestyle='--')
    ax5.legend(loc='lower left', fontsize=7, framealpha=0.9)
    ax5.set_ylim([88, 102])
    
    # (f) Path Lengths
    ax6 = fig.add_subplot(gs[1, 2])
    x = np.arange(len(obstacles))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, df_summary['mean_L_actual'], width, 
                    label='Actual', color=colors_list[0], edgecolor='black', 
                    linewidth=1, alpha=0.75)
    bars2 = ax6.bar(x + width/2, df_summary['mean_L_star'], width, 
                    label='Optimal', color='#95a5a6', edgecolor='black', 
                    linewidth=1, alpha=0.75, hatch='//')
    
    ax6.set_ylabel('Path Length (m)', fontweight='bold', fontsize=9)
    ax6.set_xlabel('Obstacles', fontweight='bold', fontsize=9)
    ax6.set_title('(f) Path Planning', fontweight='bold', fontsize=10)
    ax6.set_xticks(x)
    ax6.set_xticklabels(obstacles)
    ax6.legend(fontsize=7, framealpha=0.9, loc='upper left')
    ax6.grid(axis='y', alpha=0.25, linestyle='--')
    
    plt.suptitle('Comprehensive Performance Analysis Across Obstacle Densities', 
                 fontsize=12, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_minimal_two_figures(df_summary, df_episodes):
    """
    Alternative: 2 minimalist figures instead of 1.
    Figure 1: Main results (2×2)
    Figure 2: Supporting analysis (1×2)
    """
    
    # ============ FIGURE 1: Main Results (2×2) ============
    fig1, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig1.suptitle('Main Performance Metrics', fontsize=13, fontweight='bold', y=0.98)
    
    obstacles = df_summary['num_obstacles'].values
    colors_list = [COLORS[str(n)] for n in obstacles]
    
    # (a) Success Rate
    ax = axes[0, 0]
    bars = ax.bar(obstacles, df_summary['SR'] * 100, color=colors_list, 
                  edgecolor='black', linewidth=1.1, alpha=0.8, width=3)
    ax.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=10)
    ax.set_xlabel('Obstacles', fontweight='bold', fontsize=10)
    ax.set_title('(a) Success Rate', fontweight='bold', fontsize=11)
    ax.set_ylim([88, 105])
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.set_xticks(obstacles)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # (b) SPL
    ax = axes[0, 1]
    bars = ax.bar(obstacles, df_summary['mean_SPL'], color=colors_list, 
                  edgecolor='black', linewidth=1.1, alpha=0.8, width=3)
    ax.set_ylabel('Mean SPL', fontweight='bold', fontsize=10)
    ax.set_xlabel('Obstacles', fontweight='bold', fontsize=10)
    ax.set_title('(b) Path Efficiency (SPL)', fontweight='bold', fontsize=11)
    ax.set_ylim([0.88, 1.02])
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.set_xticks(obstacles)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # (c) Failure Rates
    ax = axes[1, 0]
    x = np.arange(len(obstacles))
    width = 0.35
    bars1 = ax.bar(x - width/2, df_summary['CR'] * 100, width, label='Collision',
                   color='#e67e22', edgecolor='black', linewidth=1, alpha=0.75)
    bars2 = ax.bar(x + width/2, df_summary['FR'] * 100, width, label='Fall',
                   color='#e74c3c', edgecolor='black', linewidth=1, alpha=0.75)
    ax.set_ylabel('Rate (%)', fontweight='bold', fontsize=10)
    ax.set_xlabel('Obstacles', fontweight='bold', fontsize=10)
    ax.set_title('(c) Failure Modes', fontweight='bold', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(obstacles)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    
    # (d) CoT
    ax = axes[1, 1]
    bars = ax.bar(obstacles, df_summary['mean_CoT'], color=colors_list, 
                  edgecolor='black', linewidth=1.1, alpha=0.8, width=3)
    ax.set_ylabel('Cost of Transport', fontweight='bold', fontsize=10)
    ax.set_xlabel('Obstacles', fontweight='bold', fontsize=10)
    ax.set_title('(d) Energy Efficiency', fontweight='bold', fontsize=11)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.set_xticks(obstacles)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.1,
                f'{h:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("fig_paper_results.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"✓ Saved: fig_paper_results.png")
    plt.close()
    
    # ============ FIGURE 2: Supporting Analysis (1×2) ============
    fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig2.suptitle('Supporting Analysis', fontsize=13, fontweight='bold', y=0.98)
    
    # (a) Performance Trends
    ax = axes[0]
    ax.plot(obstacles, df_summary['SR'] * 100, marker='o', linewidth=2.5, 
            markersize=7, label='Success Rate', color='#2E86AB', linestyle='-')
    ax.plot(obstacles, df_summary['mean_SPL'] * 100, marker='s', linewidth=2.5, 
            markersize=7, label='SPL (×100)', color='#A23B72', linestyle='--')
    ax.set_ylabel('Performance Score', fontweight='bold', fontsize=10)
    ax.set_xlabel('Obstacles', fontweight='bold', fontsize=10)
    ax.set_title('(a) Performance Degradation', fontweight='bold', fontsize=11)
    ax.set_xticks(obstacles)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_ylim([88, 102])
    
    # (b) Path Planning
    ax = axes[1]
    x = np.arange(len(obstacles))
    width = 0.35
    bars1 = ax.bar(x - width/2, df_summary['mean_L_actual'], width, 
                   label='Actual', color=colors_list[0], edgecolor='black', 
                   linewidth=1, alpha=0.75)
    bars2 = ax.bar(x + width/2, df_summary['mean_L_star'], width, 
                   label='Optimal', color='#95a5a6', edgecolor='black', 
                   linewidth=1, alpha=0.75, hatch='//')
    ax.set_ylabel('Path Length (m)', fontweight='bold', fontsize=10)
    ax.set_xlabel('Obstacles', fontweight='bold', fontsize=10)
    ax.set_title('(b) Path Planning Effectiveness', fontweight='bold', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(obstacles)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("fig_paper_analysis.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"✓ Saved: fig_paper_analysis.png")
    plt.close()


def generate_ultra_compact_figures():
    """Generate ultra-compact figures for papers."""
    print("\n" + "="*70)
    print("Generating Ultra-Compact Figures for Paper")
    print("="*70 + "\n")
    
    df_summary, df_episodes = load_data()
    
    if df_summary is None:
        print("❌ Error: eval_summary_multi_obstacles.csv not found!")
        return
    
    print("Choose your option:")
    print("  [1] ONE comprehensive figure (6 panels)")
    print("  [2] TWO minimal figures (4 + 2 panels)")
    print()
    
    # Generate both options
    print("Generating Option 1: Single comprehensive figure...")
    plot_ultra_compact_single(df_summary, df_episodes)
    
    print("\nGenerating Option 2: Two minimal figures...")
    plot_minimal_two_figures(df_summary, df_episodes)
    
    print("\n" + "="*70)
    print("✓ All ultra-compact figures generated!")
    print("="*70 + "\n")
    print("Generated files:")
    print("\nOption 1 (1 figure):")
    print("  • fig_paper_main.png - ALL metrics in single figure (6 panels)")
    print("\nOption 2 (2 figures):")
    print("  • fig_paper_results.png - Main results (4 panels)")
    print("  • fig_paper_analysis.png - Supporting analysis (2 panels)")
    print("\nRecommendation:")
    print("  • Conference papers: Use Option 1 (fig_paper_main.png)")
    print("  • Journal papers: Use Option 2 (both figures)")
    print("\nAll figures saved at 300 DPI for publication quality.")
    print("="*70 + "\n")


if __name__ == "__main__":
    generate_ultra_compact_figures()
