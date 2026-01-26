"""
Dynamic Pricing Pipeline
Run with: python pipeline.py [method]
Methods: gradient_descent, reinforcement_learning, both, compare
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.preprocessing import simple_preprocessing
from modules.util import intercept_reshaped_train_test
from modules.config import (
    LEARNING_RATE_GD, MAX_ITERATIONS, COST,
    N_PRICE_ACTIONS, LEARNING_RATE_RL, DISCOUNT,
    EPSILON, EPSILON_DECAY, EPISODES, N_EVAL_EPISODES
)
from modules.gradient_descent import gradient_descent, least_squares, APP_s, APP_d
from modules.reinforcement import PricingEnvironment, QLearningAgent, train_q_learning, evaluate_agent
from modules.naive_static import naive_static_pricing


class DynamicPricingPipeline:
    """Main pipeline for dynamic pricing experiments"""
    
    def __init__(self, save_plots=False, show_plots=True, verbose=True):
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.verbose = verbose
        self.results = {}
        
        if save_plots:
            self.output_dir = Path('output')
            self.output_dir.mkdir(exist_ok=True)
    
    def log(self, message):
        if self.verbose:
            print(message)
    
    def run_gradient_descent(self):
        self.log("\n" + "="*60)
        self.log("GRADIENT DESCENT PRICING PIPELINE")
        self.log("="*60)
        
        self.log("\n[1/7] Loading and preprocessing data...")
        preprocessor, X, y, categorical_features, numerical_features = simple_preprocessing()
        X_train, X_test, y_train, y_test = intercept_reshaped_train_test(X, y, preprocessor)
        self.log(f"  ✓ Training samples: {X_train.shape[2]}")
        self.log(f"  ✓ Test samples: {X_test.shape[2]}")
        self.log(f"  ✓ Features: {X_train.shape[1]}")
        
        self.log("\n[2/7] Training gradient descent model...")
        initial_weights = np.random.rand(X_train.shape[1])
        wh, ch = gradient_descent(
            least_squares, LEARNING_RATE_GD, MAX_ITERATIONS,
            initial_weights, X_train, y_train
        )
        index = np.argmin(ch)
        w_star = wh[index]
        self.log(f"  ✓ Converged after {index} iterations")
        
        self.log("\n[3/7] Evaluating model quality...")
        cost_train = ch[index]
        cost_test = least_squares(w_star, X_test, y_test) / X_test.shape[2]
        self.log(f"  ✓ Training Cost: {cost_train:.2f}")
        self.log(f"  ✓ Test Cost: {cost_test:.2f}")
        self.log(f"  ✓ Test/Train Ratio: {cost_test/cost_train:.3f}")
        
        self.log("\n[4/7] Computing static pricing baseline...")
        axis_2 = np.linspace(0, int(np.round(np.max(y_train), 0) + 1), 
                            int(2 * (np.round(np.max(y_train), 0) + 1)))
        app_s_train = APP_s(axis_2, y_train[0], COST)
        static_index = np.argmax(app_s_train)
        app_s_test = APP_s(axis_2, y_test[0], COST)
        app_s_star = app_s_test[static_index]
        optimal_static_price = axis_2[static_index]
        self.log(f"  ✓ Optimal Static Price: ${optimal_static_price:.2f}")
        self.log(f"  ✓ Static APP: ${app_s_star:.2f}")

        if self.show_plots or self.save_plots:
            plt.figure(figsize=(10, 6))
            plt.plot(axis_2, app_s_train)
            plt.axvline(optimal_static_price, color='r', linestyle='--', label=f'Optimal: ${optimal_static_price:.2f}')
            plt.xlabel('Price')
            plt.ylabel('APP')
            plt.title('Static Price Profit Maximization')
            plt.legend()
            plt.grid(True, alpha=0.3)
            if self.save_plots:
                plt.savefig(self.output_dir / 'static_pricing.png', dpi=300, bbox_inches='tight')
            if self.show_plots:
                plt.show()
            plt.close()
        
        if self.show_plots or self.save_plots:
            plt.figure(figsize=(10, 6))
            plt.plot(range(MAX_ITERATIONS + 1), ch)
            plt.axhline(cost_train, color='r', linestyle='--', label=f'Min Cost: {cost_train:.2f}')
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title('Gradient Descent Convergence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            if self.save_plots:
                plt.savefig(self.output_dir / 'gd_convergence.png', dpi=300, bbox_inches='tight')
            if self.show_plots:
                plt.show()
            plt.close()
        
        self.log("\n[5/7] Analyzing feature importance...")
        categorical_transformer = preprocessor.transformers_[1][1]
        feature_names_categorical = list(categorical_transformer.get_feature_names_out(
            input_features=categorical_features
        ))
        feature_names = ['Intercept'] + numerical_features + feature_names_categorical
        
        if self.verbose:
            self.log("\n  Optimized Weights:")
            for feature, weight in zip(feature_names, np.round(w_star, 2)):
                self.log(f"    {feature:30s}: {weight:6.2f}")
        
        self.log("\n[6/7] Testing dynamic pricing (no discount)...")
        app_d_1 = APP_d(w_star, X_test, y_test, COST, 1)
        self.log(f"  ✓ Dynamic APP (d=1.0): ${app_d_1:.2f}")
        
        self.log("\n[7/7] Optimizing discount factor...")
        axis_d = np.linspace(0.7, 1, 16)
        app_d = [APP_d(w_star, X_test, y_test, COST, d) for d in axis_d]
        d_star = axis_d[np.argmax(app_d)]
        app_d_star = APP_d(w_star, X_test, y_test, COST, d_star)
        self.log(f"  ✓ Optimal Discount: {d_star:.2f}")
        self.log(f"  ✓ Dynamic APP (optimized): ${app_d_star:.2f}")
        
        if self.show_plots or self.save_plots:
            plt.figure(figsize=(10, 6))
            plt.plot(axis_d, app_d, marker='o')
            plt.axvline(d_star, color='r', linestyle='--', label=f'Optimal: {d_star:.2f}')
            plt.axhline(app_d_star, color='g', linestyle='--', alpha=0.5, label=f'Max APP: ${app_d_star:.2f}')
            plt.xlabel('Discount Factor')
            plt.ylabel('APP')
            plt.title('Dynamic Price Profit Maximization')
            plt.legend()
            plt.grid(True, alpha=0.3)
            if self.save_plots:
                plt.savefig(self.output_dir / 'discount_optimization.png', dpi=300, bbox_inches='tight')
            if self.show_plots:
                plt.show()
            plt.close()
        
        self.results['gradient_descent'] = {
            'static_app': app_s_star,
            'static_price': optimal_static_price,
            'dynamic_app': app_d_star,
            'discount': d_star,
            'cost_train': cost_train,
            'cost_test': cost_test,
            'weights': dict(zip(feature_names, w_star)),
            'improvement': (app_d_star - app_s_star) / app_s_star * 100
        }
        
        self.log("\n" + "="*60)
        self.log("GRADIENT DESCENT RESULTS SUMMARY")
        self.log("="*60)
        self.log(f"Static Pricing APP:        ${app_s_star:.2f}")
        self.log(f"Dynamic Pricing APP:       ${app_d_star:.2f}")
        self.log(f"Profit Improvement:        {self.results['gradient_descent']['improvement']:.2f}%")
        self.log("="*60 + "\n")
        
        return self.results['gradient_descent']
    
    def run_reinforcement_learning(self):
        """Run Q-learning reinforcement learning"""
        self.log("\n" + "="*60)
        self.log("REINFORCEMENT LEARNING PRICING PIPELINE")
        self.log("="*60)
        
        self.log("\n[1/5] Loading and preprocessing data...")
        preprocessor, X, y, categorical_features, numerical_features = simple_preprocessing()
        X_transformed = preprocessor.fit_transform(X)
        self.log(f"  ✓ Total samples: {len(y)}")
        self.log(f"  ✓ Features: {X_transformed.shape[1]}")
        
        price_min = y.min() * 0.8
        price_max = y.max() * 1.2
        self.log(f"  ✓ Price range: ${price_min:.2f} - ${price_max:.2f}")
        
        self.log("\n[2/5] Initializing environment and agent...")
        env = PricingEnvironment(X_transformed, y, cost=0)
        agent = QLearningAgent(
            n_price_actions=N_PRICE_ACTIONS,
            price_min=price_min,
            price_max=price_max,
            learning_rate=LEARNING_RATE_RL,
            discount=DISCOUNT,
            epsilon=EPSILON,
            epsilon_decay=EPSILON_DECAY
        )
        self.log(f"  ✓ Action space: {N_PRICE_ACTIONS} discrete prices")
        self.log(f"  ✓ Learning rate: {LEARNING_RATE_RL}")
        self.log(f"  ✓ Initial epsilon: {EPSILON}")
        
        self.log(f"\n[3/5] Training Q-Learning agent ({EPISODES} episodes)...")
        rewards, profits = train_q_learning(env, agent, episodes=EPISODES)
        self.log(f"  ✓ Training complete")
        self.log(f"  ✓ Final epsilon: {agent.epsilon:.4f}")
        self.log(f"  ✓ Q-table size: {len(agent.q_table)} states")
        
        self.log(f"\n[4/5] Evaluating trained agent ({N_EVAL_EPISODES} episodes)...")
        final_app = evaluate_agent(env, agent, n_eval_episodes=N_EVAL_EPISODES)
        self.log(f"  ✓ Q-Learning APP: ${final_app:.2f}")
        
        self.log("\n[5/5] Computing static baseline...")
        optimised_static_app, optimised_price, app_s_train = naive_static_pricing(X, y, preprocessor)
        self.log(f"  ✓ Static APP: ${optimised_static_app:.2f}")
        self.log(f"  ✓ Optimal price: ${optimised_price:.2f}")
        
        if self.show_plots or self.save_plots:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            ax1.plot(rewards)
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.set_title('Q-Learning Training Progress')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(profits, label='Training APP')
            ax2.axhline(final_app, color='g', linestyle='--', label=f'Final APP: ${final_app:.2f}')
            ax2.axhline(optimised_static_app, color='r', linestyle='--', label=f'Static APP: ${optimised_static_app:.2f}')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Average Profit Per Person')
            ax2.set_title('APP Over Training')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            if self.save_plots:
                plt.savefig(self.output_dir / 'rl_training.png', dpi=300, bbox_inches='tight')
            if self.show_plots:
                plt.show()
            plt.close()
        
        self.results['reinforcement_learning'] = {
            'static_app': optimised_static_app,
            'static_price': optimised_price,
            'rl_app': final_app,
            'q_table_size': len(agent.q_table),
            'final_epsilon': agent.epsilon,
            'improvement': (final_app - optimised_static_app) / optimised_static_app * 100
        }
        
        self.log("\n" + "="*60)
        self.log("REINFORCEMENT LEARNING RESULTS SUMMARY")
        self.log("="*60)
        self.log(f"Static Pricing APP:        ${optimised_static_app:.2f}")
        self.log(f"Q-Learning APP:            ${final_app:.2f}")
        self.log(f"Profit Improvement:        {self.results['reinforcement_learning']['improvement']:.2f}%")
        self.log("="*60 + "\n")
        
        return self.results['reinforcement_learning']
    
    def run_comparison(self):
        """Run both methods and compare"""
        gd_results = self.run_gradient_descent()
        rl_results = self.run_reinforcement_learning()
        
        self.log("\n" + "="*60)
        self.log("FINAL COMPARISON: GRADIENT DESCENT vs Q-LEARNING")
        self.log("="*60)
        self.log(f"\nStatic Baseline:           ${gd_results['static_app']:.2f}")
        self.log(f"Gradient Descent:          ${gd_results['dynamic_app']:.2f} (+{gd_results['improvement']:.2f}%)")
        self.log(f"Q-Learning:                ${rl_results['rl_app']:.2f} (+{rl_results['improvement']:.2f}%)")
        
        winner = "Q-Learning" if rl_results['rl_app'] > gd_results['dynamic_app'] else "Gradient Descent"
        self.log(f"\nBest model: {winner}")
        self.log("="*60 + "\n")
        
        return self.results


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description='Dynamic Pricing Pipeline for Uber Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py gradient_descent              # Run gradient descent only
  python pipeline.py reinforcement_learning        # Run Q-learning only
  python pipeline.py compare                       # Run both and compare
  python pipeline.py compare --no-plots            # Run without showing plots
  python pipeline.py compare --save-plots          # Save plots to output/
  python pipeline.py gradient_descent --quiet      # Minimal output
        """
    )
    
    parser.add_argument(
        'method',
        choices=['gradient_descent', 'reinforcement_learning', 'compare', 'both'],
        help='Method to run'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot display'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save plots to output/ directory'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )
    
    args = parser.parse_args()
    
    pipeline = DynamicPricingPipeline(
        save_plots=args.save_plots,
        show_plots=not args.no_plots,
        verbose=not args.quiet
    )
    
    try:
        if args.method == 'gradient_descent':
            pipeline.run_gradient_descent()
        elif args.method == 'reinforcement_learning':
            pipeline.run_reinforcement_learning()
        elif args.method in ['compare', 'both']:
            pipeline.run_comparison()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()