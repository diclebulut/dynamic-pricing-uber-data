
# ============================================
# Data Path
# ============================================

DATA_PATH = 'src/data/uber_data.csv'

# ============================================
# Columns as features
# ============================================

CATEGORICAL_FEATURES = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']
NUMERICAL_FEATURES = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 'Expected_Ride_Duration']
TARGET_COLUMN = 'Historical_Cost_of_Ride'

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
COST = 0

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