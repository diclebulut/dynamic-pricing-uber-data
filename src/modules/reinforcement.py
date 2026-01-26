
import numpy as np

class PricingEnvironment:
    def __init__(self, features, true_prices, cost=0):
        """
        features: Customer feature matrix (N x D)
        true_prices: Historical prices (willingness to pay) (N,)
        cost: Operating cost per ride
        """
        self.features = features
        self.true_prices = true_prices
        self.cost = cost
        self.n_customers = len(features)
        self.current_customer = 0
        
    def reset(self):
        """Start with a random customer"""
        self.current_customer = np.random.randint(0, self.n_customers)
        return self._get_state()
    
    def _get_state(self):
        """Return current customer features"""
        return self.features[self.current_customer]
    
    def step(self, price):
        """
        Agent proposes a price, customer accepts/rejects
        
        Returns:
            state: Next customer's features
            reward: Profit if accepted, 0 if rejected
            done: True (episodic task - one price per customer)
        """
        willingness = self.true_prices[self.current_customer]
        
        if price <= willingness:
            reward = price - self.cost  
        else:
            reward = 0  
        
        self.current_customer = (self.current_customer + 1) % self.n_customers
        next_state = self._get_state()
        done = True  
        
        return next_state, reward, done
    

class QLearningAgent:
    def __init__(self, n_price_actions=20, price_min=5, price_max=50, 
                 learning_rate=0.1, discount=0.95, epsilon=1.0, epsilon_decay=0.995):
        """
        n_price_actions: Number of discrete price points
        price_min/max: Price range
        """
        self.price_actions = np.linspace(price_min, price_max, n_price_actions)
        self.n_actions = n_price_actions
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        self.q_table = {}
    
    def _discretize_state(self, state):
        """Convert continuous state to discrete hash"""
        discretized = tuple(np.round(state, 1))
        return discretized
    
    def _ensure_state_exists(self, state):
        """Initialize Q-values for new states"""
        state_key = self._discretize_state(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return state_key
    
    def get_q_values(self, state):
        """Get Q-values for a state"""
        state_key = self._ensure_state_exists(state)
        return self.q_table[state_key]
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        self._ensure_state_exists(state)
        
        if training and np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.n_actions)
        else:
            q_values = self.get_q_values(state)
            action_idx = np.argmax(q_values)
        
        return action_idx, self.price_actions[action_idx]
    
    def update(self, state, action_idx, reward, next_state, done):
        """Q-learning update rule"""
        state_key = self._ensure_state_exists(state)
        self._ensure_state_exists(next_state)
        
        q_current = self.q_table[state_key][action_idx]
        
        if done:
            q_target = reward
        else:
            q_next_max = np.max(self.get_q_values(next_state))
            q_target = reward + self.gamma * q_next_max
        
        self.q_table[state_key][action_idx] += self.lr * (q_target - q_current)
    
    def decay_epsilon(self):
        """Reduce exploration over time"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_q_learning(env, agent, episodes=1000, verbose=True):
    """Train Q-learning agent"""
    episode_rewards = []
    episode_profits = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for _ in range(env.n_customers):
            action_idx, price = agent.select_action(state, training=True)
            
            next_state, reward, done = env.step(price)
            
            agent.update(state, action_idx, reward, next_state, done)
            
            total_reward += reward
            state = next_state
        
        agent.decay_epsilon()
        
        episode_rewards.append(total_reward)
        episode_profits.append(total_reward / env.n_customers)  # APP
        
        if verbose and (episode + 1) % 100 == 0:
            avg_profit = np.mean(episode_profits[-100:])
            print(f"Episode {episode + 1}/{episodes} | Avg APP: {avg_profit:.2f} | Epsilon: {agent.epsilon:.3f}")
    
    return episode_rewards, episode_profits

def evaluate_agent(env, agent, n_eval_episodes=10):
    """Test trained agent"""
    total_profits = []
    
    for _ in range(n_eval_episodes):
        state = env.reset()
        episode_profit = 0
        
        for _ in range(env.n_customers):
            action_idx, price = agent.select_action(state, training=False)
            next_state, reward, done = env.step(price)
            episode_profit += reward
            state = next_state
        
        total_profits.append(episode_profit / env.n_customers)
    
    return np.mean(total_profits)