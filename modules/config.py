
# ============================================
# Data Path
# ============================================

DATA_PATH = 'data/uber_data.csv'

# ============================================
# Basic train test split parameters
# ============================================

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================
# Gradient Descent parameters
# ============================================

LEARNING_RATE_GD = 1e-5
MAX_ITERATIONS = 50

# ============================================
# Reinforcement Learning parameters
# ============================================

N_PRICE_ACTIONS=20
LEARNING_RATE_RL=0.1
DISCOUNT=0.95
EPSILON=1.0
EPSILON_DECAY=0.995
EPISODES=1000
N_EVAL_EPISODES=50