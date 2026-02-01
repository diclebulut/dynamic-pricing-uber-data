"""
Main Pipeline for Dynamic Pricing Comparison

Orchestrates the full workflow: data preparation, static pricing,
DQN dynamic pricing, and comparative analysis with optional output.
"""

import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from modules.preprocessing import prepare_columns
from modules.config import TAXI_ZONE_PATH, NUM_PRICE_ACTIONS, NUM_TRAINING_EPISODES, BATCH_SIZE, TARGET_UPDATE_FREQUENCY, NUM_TEST_EPISODES, STEPS_PER_TEST_EPISODE, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_FEATURE
from modules.static import train_static_model, apply_static_pricing, get_static_metrics
from modules.reinforcement import (
    PricingEnvironment,
    DQNAgent,
    train_dqn_agent,
    test_dqn_agent,
    get_dqn_metrics
)


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def prepare_feature_engineered_data(
    train_data_raw: pd.DataFrame,
    test_data_raw: pd.DataFrame,
    numerical_features: list = None,
    categorical_features: list = None
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int, StandardScaler, OneHotEncoder]:
    """
    Prepare feature-engineered data for model training.
    
    Args:
        train_data_raw: Raw training data
        test_data_raw: Raw test data
        numerical_features: List of numerical feature columns
        categorical_features: List of categorical feature columns
    
    Returns:
        tuple: (train_data, test_data, state_size, action_size, scaler, encoder)
    """
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES
    
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES
    
    target_feature = TARGET_FEATURE
    
    print("="*70)
    print("FEATURE ENGINEERING CONFIGURATION")
    print("="*70)
    print(f"Numerical Features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical Features ({len(categorical_features)}): {categorical_features}")
    print(f"Target Feature: {target_feature}\n")
    
    scaler = StandardScaler()
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    
    print("="*70)
    print("FEATURE ENGINEERING (FIT ON TRAINING DATA)")
    print("="*70)
    
    print("Fitting and scaling numerical features on training data...")
    scaler.fit(train_data_raw[numerical_features])
    train_numerical_data = scaler.transform(train_data_raw[numerical_features])
    test_numerical_data = scaler.transform(test_data_raw[numerical_features])
    
    train_numerical_df = pd.DataFrame(
        train_numerical_data,
        columns=numerical_features,
        index=train_data_raw.index
    )
    
    test_numerical_df = pd.DataFrame(
        test_numerical_data,
        columns=numerical_features,
        index=test_data_raw.index
    )
    
    print("Fitting and one-hot encoding categorical features on training data...")
    one_hot_encoder.fit(train_data_raw[categorical_features])
    train_categorical_data = one_hot_encoder.transform(train_data_raw[categorical_features])
    test_categorical_data = one_hot_encoder.transform(test_data_raw[categorical_features])
    
    categorical_feature_names = list(one_hot_encoder.get_feature_names_out(categorical_features))
    
    train_categorical_df = pd.DataFrame(
        train_categorical_data,
        columns=categorical_feature_names,
        index=train_data_raw.index
    )
    
    test_categorical_df = pd.DataFrame(
        test_categorical_data,
        columns=categorical_feature_names,
        index=test_data_raw.index
    )
    
    # Combine features
    train_data = pd.concat([train_numerical_df, train_categorical_df], axis=1)
    train_data[target_feature] = train_data_raw[target_feature].values
    
    test_data = pd.concat([test_numerical_df, test_categorical_df], axis=1)
    test_data[target_feature] = test_data_raw[target_feature].values
    
    state_size = len(train_data.columns) - 1
    action_size = NUM_PRICE_ACTIONS
    
    print(f"\nProcessed Training Data Shape: {train_data.shape}")
    print(f"Processed Testing Data Shape: {test_data.shape}")
    print(f"State Size: {state_size}")
    print(f"Action Size: {action_size}\n")
    
    return train_data, test_data, state_size, action_size, scaler, one_hot_encoder


def run_static_pricing_pipeline(
    train_data_raw: pd.DataFrame,
    test_data_raw: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run static pricing strategy.
    
    Args:
        train_data_raw: Training data
        test_data_raw: Test data
        verbose: Print detailed output
    
    Returns:
        pd.DataFrame: Results with static pricing applied
    """
    print("\n" + "="*70)
    print("STATIC PRICING STRATEGY (CALIBRATED ON TRAINING DATA)")
    print("="*70)
    
    price_per_mile = train_static_model(train_data_raw)
    print(f"\n1. Calibration Phase (using TRAINING data):")
    print(f"   Calibrated Price Per Mile: ${price_per_mile:.2f}")
    
    print(f"\n2. Testing Phase (applying to TEST data):")
    static_results = apply_static_pricing(test_data_raw, price_per_mile, verbose=verbose)
    
    return static_results


def run_dqn_pipeline(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    state_size: int,
    action_size: int,
    num_episodes: int = None,
    batch_size: int = None,
    update_frequency: int = None,
    test_episodes: int = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run DQN dynamic pricing strategy.
    
    Args:
        train_data: Feature-engineered training data
        test_data: Feature-engineered test data
        state_size: State vector size
        action_size: Number of pricing actions
        num_episodes: Training episodes
        batch_size: Experience replay batch size
        update_frequency: Target network update frequency
        test_episodes: Number of test episodes
        verbose: Print detailed output
    
    Returns:
        pd.DataFrame: Test results with DQN pricing
    """
    if num_episodes is None:
        num_episodes = NUM_TRAINING_EPISODES
    if batch_size is None:
        batch_size = BATCH_SIZE
    if update_frequency is None:
        update_frequency = TARGET_UPDATE_FREQUENCY
    if test_episodes is None:
        test_episodes = NUM_TEST_EPISODES
    env_train = PricingEnvironment(train_data, num_price_actions=action_size)
    env_test = PricingEnvironment(test_data, num_price_actions=action_size)
    
    agent = DQNAgent(state_size, action_size, learning_rate=0.001)
    
    train_dqn_agent(
        env_train,
        agent,
        num_episodes=num_episodes,
        batch_size=batch_size,
        update_frequency=update_frequency,
        verbose=verbose
    )
    
    dqn_results = test_dqn_agent(
        env_test,
        agent,
        num_episodes=test_episodes,
        steps_per_episode=STEPS_PER_TEST_EPISODE,
        verbose=verbose
    )
    
    return dqn_results


def compare_strategies(
    static_results: pd.DataFrame,
    dqn_results: pd.DataFrame,
    test_data_raw: pd.DataFrame,
    verbose: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """
    Compare static and DQN strategies.
    
    Args:
        static_results: Results from static pricing
        dqn_results: Results from DQN pricing
        test_data_raw: Original test data for reference
        verbose: Print detailed output
    
    Returns:
        tuple: (static_metrics, dqn_metrics, comparison_df)
    """
    if verbose:
        print("\n" + "="*70)
        print("COMPARISON: DYNAMIC PRICING (DQN) vs STATIC PRICING")
        print("="*70)
    
    static_metrics = get_static_metrics(static_results)
    dqn_metrics = get_dqn_metrics(dqn_results)
    
    if verbose:
        print("\nSTATIC PRICING METRICS:")
        print(f"  Average Fare Per Ride: ${static_metrics['avg_price']:.2f}")
        print(f"  Total Revenue: ${static_metrics['total_revenue']:.2f}")
        print(f"  Average Revenue Per Ride: ${static_metrics['avg_revenue_per_ride']:.2f}")
        print(f"  Acceptance Rate: {static_metrics['acceptance_rate']:.2%}")
        print(f"  Accepted Rides: {static_metrics['accepted_rides']} / {static_metrics['total_rides']}")
        
        print("\nDQN DYNAMIC PRICING METRICS:")
        print(f"  Average Expected Revenue Multiplier: {dqn_metrics['avg_reward']:.4f}x")
        print(f"  Average Acceptance Rate: {dqn_metrics['acceptance_rate']:.2%}")
        print(f"  Total Revenue: ${dqn_metrics['total_revenue']:.2f}")
        print(f"  Average Revenue per Ride: ${dqn_metrics['avg_revenue_per_ride']:.2f}")
        print(f"  Accepted Rides: {dqn_metrics['accepted_rides']} ({dqn_metrics['accepted_rides']/dqn_metrics['total_rides']:.2%})")
        
        revenue_diff = dqn_metrics['total_revenue'] - static_metrics['total_revenue']
        revenue_improvement = (revenue_diff / static_metrics['total_revenue'] * 100) if static_metrics['total_revenue'] > 0 else 0
        
        print("\nREVENUE COMPARISON:")
        print(f"  Static Total Revenue: ${static_metrics['total_revenue']:.2f}")
        print(f"  DQN Total Revenue: ${dqn_metrics['total_revenue']:.2f}")
        print(f"  Revenue Difference: ${revenue_diff:.2f}")
        print(f"  Improvement: {revenue_improvement:+.2f}%")
    
    comparison_df = pd.DataFrame({
        'dqn_price': dqn_results['predicted_price'].values,
        'dqn_accepted': dqn_results['is_accepted'].values,
        'dqn_multiplier': dqn_results['multiplier'].values,
        'dqn_reward': dqn_results['reward'].values,
    })
    
    return static_metrics, dqn_metrics, comparison_df


def generate_visualizations(
    static_results: pd.DataFrame,
    dqn_results: pd.DataFrame,
    output_dir: Path = None
) -> Path:
    """
    Generate comparison visualizations.
    
    Args:
        static_results: Static pricing results
        dqn_results: DQN pricing results
        output_dir: Directory to save plots
    
    Returns:
        Path: Directory where plots were saved
    """
    if output_dir is None:
        output_dir = Path('outputs/visualizations')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    ax1 = axes[0, 0]
    ax1.scatter(dqn_results['multiplier'], dqn_results['acceptance_rate'],
                alpha=0.5, s=50, c=dqn_results['is_accepted'], cmap='RdYlGn', edgecolors='black')
    ax1.set_xlabel('Price Multiplier', fontsize=12)
    ax1.set_ylabel('Acceptance Rate', fontsize=12)
    ax1.set_title('Acceptance Rate vs Price Multiplier', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.hist(dqn_results['reward'], bins=30, color='#F18F01', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Reward', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Rewards (Testing)', fontsize=14, fontweight='bold')
    ax2.axvline(np.mean(dqn_results['reward']), color='red', linestyle='--', linewidth=2)
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3 = axes[1, 0]
    ax3.hist(dqn_results['predicted_price'], bins=30, color='#06A77D', alpha=0.7,
             label='DQN Price', edgecolor='black')
    ax3.hist(dqn_results['actual_fare'], bins=30, color='#D62828', alpha=0.5,
             label='Actual Fare', edgecolor='black')
    ax3.set_xlabel('Price ($)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Predicted Price vs Actual Fare Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = axes[1, 1]
    acceptance_data = [
        dqn_results['is_accepted'].sum(),
        len(dqn_results) - dqn_results['is_accepted'].sum()
    ]
    colors = ['#2E86AB', '#A23B72']
    ax4.pie(acceptance_data, labels=['Accepted', 'Rejected'], autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax4.set_title('DQN Acceptance Rate Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plot_path = output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_dir


def save_markdown_report(
    static_metrics: Dict[str, Any],
    dqn_metrics: Dict[str, Any],
    comparison_df: pd.DataFrame,
    output_dir: Path = None,
    complexity: str = 'full'
) -> Path:
    """
    Save analysis results as markdown report.
    
    Args:
        static_metrics: Static pricing metrics
        dqn_metrics: DQN pricing metrics
        comparison_df: Comparison dataframe
        output_dir: Output directory
        complexity: 'summary', 'standard', or 'full'
    
    Returns:
        Path: Path to saved markdown file
    """
    if output_dir is None:
        output_dir = Path('outputs/reports')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_path = output_dir / f"pricing_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_path, 'w') as f:
        f.write("# Dynamic Pricing Analysis Report\n\n")
        f.write(f"**Generated:** {timestamp}\n\n")
        
        f.write("## Executive Summary\n\n")
        revenue_diff = dqn_metrics['total_revenue'] - static_metrics['total_revenue']
        revenue_improvement = (revenue_diff / static_metrics['total_revenue'] * 100) if static_metrics['total_revenue'] > 0 else 0
        
        f.write(f"- **Revenue Improvement:** {revenue_improvement:+.2f}%\n")
        f.write(f"- **DQN Acceptance Rate:** {dqn_metrics['acceptance_rate']:.2%}\n")
        f.write(f"- **Static Acceptance Rate:** {static_metrics['acceptance_rate']:.2%}\n\n")
        
        f.write("## Static Pricing Results\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Average Price Per Ride | ${static_metrics['avg_price']:.2f} |\n")
        f.write(f"| Total Revenue | ${static_metrics['total_revenue']:.2f} |\n")
        f.write(f"| Acceptance Rate | {static_metrics['acceptance_rate']:.2%} |\n")
        f.write(f"| Accepted Rides | {static_metrics['accepted_rides']} / {static_metrics['total_rides']} |\n\n")
        
        f.write("## DQN Dynamic Pricing Results\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Average Price Per Ride | ${dqn_metrics['avg_price']:.2f} |\n")
        f.write(f"| Average Revenue Multiplier | {dqn_metrics['avg_reward']:.4f}x |\n")
        f.write(f"| Total Revenue | ${dqn_metrics['total_revenue']:.2f} |\n")
        f.write(f"| Acceptance Rate | {dqn_metrics['acceptance_rate']:.2%} |\n")
        f.write(f"| Accepted Rides | {dqn_metrics['accepted_rides']} / {dqn_metrics['total_rides']} |\n\n")
        
        if complexity in ['standard', 'full']:
            f.write("## Detailed Comparison\n\n")
            f.write(f"- **Revenue Difference:** ${revenue_diff:.2f}\n")
            f.write(f"- **DQN Price Std Dev:** ${dqn_metrics['price_std']:.2f}\n")
            f.write(f"- **Static Price Std Dev:** ${static_metrics['price_std']:.2f}\n\n")
        
        if complexity == 'full':
            f.write("## Acceptance Rate Comparison\n\n")
            acceptance_improvement = (dqn_metrics['acceptance_rate'] - static_metrics['acceptance_rate']) * 100
            f.write(f"- **Acceptance Rate Difference:** {acceptance_improvement:+.2f} percentage points\n")
            f.write(f"- **DQN Acceptance:** {dqn_metrics['accepted_rides']} rides\n")
            f.write(f"- **Static Acceptance:** {static_metrics['accepted_rides']} rides\n\n")
            
            f.write("## Statistical Summary\n\n")
            f.write(f"- **DQN Avg Multiplier:** {dqn_metrics['avg_multiplier']:.4f}x\n")
            f.write(f"- **Total Rides Evaluated:** {len(comparison_df)}\n\n")
    
    return report_path


def main(
    verbose: bool = True,
    save_markdown: bool = False,
    markdown_complexity: str = 'full',
    save_plots: bool = False,
    num_training_episodes: int = None,
    batch_size: int = None,
    update_frequency: int = None,
    test_episodes: int = None
):
    """
    Main pipeline orchestration.
    
    Args:
        verbose: Print detailed output
        save_markdown: Save results as markdown report
        markdown_complexity: 'summary', 'standard', or 'full'
        save_plots: Save visualization plots
        num_training_episodes: DQN training episodes
        batch_size: Experience replay batch size
        update_frequency: Target network update frequency
        test_episodes: Number of test episodes
    """
    if num_training_episodes is None:
        num_training_episodes = NUM_TRAINING_EPISODES
    if batch_size is None:
        batch_size = BATCH_SIZE
    if update_frequency is None:
        update_frequency = TARGET_UPDATE_FREQUENCY
    if test_episodes is None:
        test_episodes = NUM_TEST_EPISODES
    set_seeds(42)
    
    if verbose:
        print("\n" + "="*70)
        print("DYNAMIC PRICING ANALYSIS PIPELINE")
        print("="*70)
    
    if verbose:
        print("\nStep 1: Loading and preparing data...")
    
    uber_trips = prepare_columns()
    data_raw = uber_trips.copy()
    
    train_data_raw, test_data_raw = train_test_split(
        data_raw,
        test_size=0.2,
        random_state=42
    )
    
    if verbose:
        print(f"  Training data: {len(train_data_raw)} rides")
        print(f"  Test data: {len(test_data_raw)} rides")
    
    if verbose:
        print("\nStep 2: Feature engineering...")
    
    train_data, test_data, state_size, action_size, scaler, encoder = prepare_feature_engineered_data(
        train_data_raw,
        test_data_raw
    )
    
    if verbose:
        print("\nStep 3: Running static pricing strategy...")
    
    static_results = run_static_pricing_pipeline(
        train_data_raw,
        test_data_raw,
        verbose=verbose
    )
    
    if verbose:
        print("\nStep 4: Running DQN dynamic pricing strategy...")
    
    dqn_results = run_dqn_pipeline(
        train_data,
        test_data,
        state_size,
        action_size,
        num_episodes=num_training_episodes,
        batch_size=batch_size,
        update_frequency=update_frequency,
        test_episodes=test_episodes,
        verbose=verbose
    )
    
    if verbose:
        print("\nStep 5: Comparing strategies...")
    
    static_metrics, dqn_metrics, comparison_df = compare_strategies(
        static_results,
        dqn_results,
        test_data_raw,
        verbose=verbose
    )
    
    if save_markdown or save_plots:
        if verbose:
            print("\nStep 6: Saving outputs...")
        
        if save_markdown:
            report_path = save_markdown_report(
                static_metrics,
                dqn_metrics,
                comparison_df,
                complexity=markdown_complexity
            )
            if verbose:
                print(f"  ✓ Markdown report saved to {report_path}")
        
        if save_plots:
            plot_dir = generate_visualizations(
                static_results,
                dqn_results
            )
            if verbose:
                print(f"  ✓ Plots saved to {plot_dir}")
    
    if verbose:
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70 + "\n")
    
    return {
        'static_metrics': static_metrics,
        'dqn_metrics': dqn_metrics,
        'comparison_df': comparison_df,
        'train_data': train_data,
        'test_data': test_data,
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Dynamic Pricing Analysis Pipeline')
    parser.add_argument('--verbose', type=bool, default=True, help='Print detailed output')
    parser.add_argument('--save-markdown', action='store_true', help='Save markdown report')
    parser.add_argument('--markdown-complexity', choices=['summary', 'standard', 'full'],
                       default='full', help='Markdown report complexity level')
    parser.add_argument('--save-plots', action='store_true', help='Save visualization plots')
    parser.add_argument('--episodes', type=int, default=100, help='DQN training episodes')
    parser.add_argument('--batch-size', type=int, default=32, help='Experience replay batch size')
    parser.add_argument('--update-freq', type=int, default=10, help='Target network update frequency')
    parser.add_argument('--test-episodes', type=int, default=20, help='Number of test episodes')
    
    args = parser.parse_args()
    
    main(
        verbose=args.verbose,
        save_markdown=args.save_markdown,
        markdown_complexity=args.markdown_complexity,
        save_plots=args.save_plots,
        num_training_episodes=args.episodes,
        batch_size=args.batch_size,
        update_frequency=args.update_freq,
        test_episodes=args.test_episodes
    )
