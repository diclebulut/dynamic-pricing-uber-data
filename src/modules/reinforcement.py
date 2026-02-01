"""
Deep Q-Network (DQN) Reinforcement Learning Module

Implements a DQN agent for dynamic pricing optimization with
experience replay and target network architecture.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, Dict, List, Any
from .config import (
    DQN_HIDDEN_LAYER_1, DQN_HIDDEN_LAYER_2, DQN_HIDDEN_LAYER_3,
    NUM_PRICE_ACTIONS, PRICE_MULTIPLIER_MIN, PRICE_MULTIPLIER_MAX,
    MAX_ACCEPTABLE_PRICE_DIFF_RATIO, SIGMOID_SCALING_FACTOR_RATIO,
    LEARNING_RATE, GAMMA, EPSILON_INITIAL, EPSILON_MIN, EPSILON_DECAY,
    MEMORY_BUFFER_SIZE, NUM_TRAINING_EPISODES, BATCH_SIZE, TARGET_UPDATE_FREQUENCY
)


class DQN(nn.Module):
    """Deep Q-Network architecture for pricing decisions"""
    
    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, DQN_HIDDEN_LAYER_1)
        self.fc2 = nn.Linear(DQN_HIDDEN_LAYER_1, DQN_HIDDEN_LAYER_2)
        self.fc3 = nn.Linear(DQN_HIDDEN_LAYER_2, DQN_HIDDEN_LAYER_3)
        self.fc4 = nn.Linear(DQN_HIDDEN_LAYER_3, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class PricingEnvironment:
    """Environment that simulates pricing decisions with acceptance rates"""
    
    def __init__(self, data: pd.DataFrame, num_price_actions: int = NUM_PRICE_ACTIONS, target_column: str = 'final_fare'):
        self.data = data.reset_index(drop=True)
        self.num_price_actions = num_price_actions
        self.current_idx = 0
        self.target_column = target_column
        self.state_columns = [col for col in data.columns if col != target_column]
        
        # Price multipliers from PRICE_MULTIPLIER_MIN to PRICE_MULTIPLIER_MAX
        self.price_multipliers = np.linspace(PRICE_MULTIPLIER_MIN, PRICE_MULTIPLIER_MAX, num_price_actions)
        
    def reset(self):
        """Reset environment to random trip"""
        self.current_idx = random.randint(0, len(self.data) - 1)
        return self._get_state()
    
    def _get_state(self):
        """Get current state (trip features)"""
        trip = self.data.iloc[self.current_idx]
        state = np.array([trip[col] for col in self.state_columns], dtype=np.float32)
        return state
    
    @staticmethod
    def calculate_acceptance_rate(predicted_price: float, actual_fare: float) -> float:
        """Calculate acceptance probability based on price difference"""
        if actual_fare <= 0:
            return 0.0
        
        price_diff = abs(predicted_price - actual_fare)
        max_acceptable_diff = actual_fare * MAX_ACCEPTABLE_PRICE_DIFF_RATIO
        
        acceptance_rate = 1.0 / (1.0 + np.exp((price_diff - max_acceptable_diff) / (max_acceptable_diff * SIGMOID_SCALING_FACTOR_RATIO)))
        return acceptance_rate
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and return reward"""
        trip = self.data.iloc[self.current_idx]
        actual_fare = trip[self.target_column]
        
        multiplier = self.price_multipliers[action]
        predicted_price = actual_fare * multiplier
        
        acceptance_rate = self.calculate_acceptance_rate(predicted_price, actual_fare)
        is_accepted = np.random.rand() < acceptance_rate
        
        reward = predicted_price / actual_fare if is_accepted else 0.0
        
        current_idx_before = self.current_idx
        self.current_idx = (self.current_idx + 1) % len(self.data)
        
        next_state = self._get_state()
        
        return next_state, reward, False, {
            'predicted_price': predicted_price,
            'actual_fare': actual_fare,
            'acceptance_rate': acceptance_rate,
            'is_accepted': is_accepted,
            'multiplier': multiplier,
            'ride_index': current_idx_before
        }


class DQNAgent:
    """DQN Agent with experience replay and epsilon-greedy exploration"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = LEARNING_RATE):
        self.state_size = state_size
        self.action_size = action_size
        
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.gamma = GAMMA
        self.epsilon = EPSILON_INITIAL
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.memory = deque(maxlen=MEMORY_BUFFER_SIZE)
        
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return np.argmax(q_values.cpu().numpy()[0])
    
    def replay(self, batch_size: int):
        """Train on minibatch from memory"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.array([x[0] for x in minibatch]))
        actions = torch.LongTensor(np.array([x[1] for x in minibatch]))
        rewards = torch.FloatTensor(np.array([x[2] for x in minibatch]))
        next_states = torch.FloatTensor(np.array([x[3] for x in minibatch]))
        dones = torch.FloatTensor(np.array([x[4] for x in minibatch]))
        
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.criterion(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        """Update target network"""
        self.target_model.load_state_dict(self.model.state_dict())


def train_dqn_agent(
    env: PricingEnvironment,
    agent: DQNAgent,
    num_episodes: int = None,
    batch_size: int = None,
    update_frequency: int = None,
    verbose: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Train DQN agent on pricing environment.
    
    Args:
        env: Pricing environment
        agent: DQN agent
        num_episodes: Number of training episodes
        batch_size: Experience replay batch size
        update_frequency: Update target network every N episodes
        verbose: Print training progress
    
    Returns:
        tuple: (episode_rewards, episode_acceptances)
    """
    if num_episodes is None:
        num_episodes = NUM_TRAINING_EPISODES
    if batch_size is None:
        batch_size = BATCH_SIZE
    if update_frequency is None:
        update_frequency = TARGET_UPDATE_FREQUENCY
    
    episode_rewards = []
    episode_acceptances = []
    
    if verbose:
        print("\n" + "="*70)
        print("DQN TRAINING FOR DYNAMIC PRICING")
        print("="*70)
        print(f"Episodes: {num_episodes}")
        print(f"Batch Size: {batch_size}")
        print(f"Update Frequency: {update_frequency}")
        print(f"Initial Epsilon: {agent.epsilon}")
        print(f"Discount Factor (Gamma): {agent.gamma}\n")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_accepted = 0
        episode_steps = min(len(env.data), 100)
        
        for step in range(episode_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay(batch_size)
            
            episode_reward += reward
            episode_accepted += int(info['is_accepted'])
            state = next_state
        
        if (episode + 1) % update_frequency == 0:
            agent.update_target_model()
        
        episode_rewards.append(episode_reward / episode_steps)
        episode_acceptances.append(episode_accepted / episode_steps)
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:3d}/{num_episodes} | "
                  f"Avg Reward: {episode_rewards[-1]:7.4f} | "
                  f"Acceptance Rate: {episode_acceptances[-1]:6.2%} | "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    if verbose:
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
    
    return episode_rewards, episode_acceptances


def test_dqn_agent(
    env: PricingEnvironment,
    agent: DQNAgent,
    num_episodes: int = 20,
    steps_per_episode: int = 100,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Test trained DQN agent.
    
    Args:
        env: Pricing environment
        agent: Trained DQN agent
        num_episodes: Number of test episodes
        steps_per_episode: Steps per episode
        verbose: Print test results
    
    Returns:
        pd.DataFrame: Detailed test results
    """
    agent.epsilon = 0#greedy
    test_details = []
    
    for episode in range(num_episodes):
        state = env.reset()
        
        for step in range(steps_per_episode):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            test_details.append({
                'ride_index': info['ride_index'],
                'predicted_price': info['predicted_price'],
                'actual_fare': info['actual_fare'],
                'acceptance_rate': info['acceptance_rate'],
                'is_accepted': info['is_accepted'],
                'multiplier': info['multiplier'],
                'reward': reward
            })
            
            state = next_state
    
    results_df = pd.DataFrame(test_details)
    
    if verbose:
        print("\n" + "="*70)
        print("TESTING TRAINED MODEL")
        print("="*70)
        print(f"\nTest Results (Average over {num_episodes} episodes):")
        print(f"  Average Reward: {results_df['reward'].mean():.4f}")
        print(f"  Avg Acceptance Rate: {results_df['is_accepted'].mean():.2%}")
        print(f"  Avg Revenue Multiplier: {results_df['multiplier'].mean():.4f}x")
        print(f"  Min Multiplier: {results_df['multiplier'].min():.4f}x")
        print(f"  Max Multiplier: {results_df['multiplier'].max():.4f}x")
        print(f"\nPrice Statistics:")
        print(f"  Average Predicted Price: ${results_df['predicted_price'].mean():.2f}")
        print(f"  Min Predicted Price: ${results_df['predicted_price'].min():.2f}")
        print(f"  Max Predicted Price: ${results_df['predicted_price'].max():.2f}")
        print(f"  Total Accepted Rides: {results_df['is_accepted'].sum()}")
    
    return results_df


def get_dqn_metrics(results: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract key metrics from DQN test results.
    
    Args:
        results: Output from test_dqn_agent()
    
    Returns:
        dict: Dictionary of performance metrics
    """
    accepted_rides = results['is_accepted'].sum()
    total_revenue = results[results['is_accepted']]['predicted_price'].sum()
    
    return {
        'avg_reward': results['reward'].mean(),
        'acceptance_rate': results['is_accepted'].mean(),
        'accepted_rides': accepted_rides,
        'total_rides': len(results),
        'total_revenue': total_revenue,
        'avg_revenue_per_ride': total_revenue / len(results),
        'avg_multiplier': results['multiplier'].mean(),
        'price_std': results['predicted_price'].std(),
        'avg_price': results['predicted_price'].mean(),
    }
