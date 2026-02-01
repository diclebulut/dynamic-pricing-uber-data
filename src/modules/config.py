# ============================================
# Data Paths
# ============================================

DATA_PATH = 'data/fhvhv_tripdata_2021-01.parquet'
TAXI_ZONE_PATH = 'data/taxi_zone_lookup.csv'
WEATHER_NYC_PATH = 'data/nyc 2021-01-01 to 2021-12-31.csv'

# ============================================
# Columns as features
# ============================================
NUMERICAL_FEATURES = ['trip_miles', 'tips', 'wait', 'day_of_month', 'precip']
CATEGORICAL_FEATURES = ['Pickup_Borough', 'Dropoff_Borough', 'preciptype',
                               'time_of_day', 'shared_request_flag', 'shared_match_flag']
TARGET_FEATURE = 'final_fare'


# ============================================
# Reinforcement Learning - DQN Configuration
# ============================================

#NN Architecture
DQN_HIDDEN_LAYER_1 = 256
DQN_HIDDEN_LAYER_2 = 256
DQN_HIDDEN_LAYER_3 = 128

# Pricing Environment
NUM_PRICE_ACTIONS = 11  #n of discrete pricing actions
PRICE_MULTIPLIER_MIN = 0.8  #min price multiplier
PRICE_MULTIPLIER_MAX = 1.2  #max price multiplier
MAX_ACCEPTABLE_PRICE_DIFF_RATIO = 0.5  #price diff threshold (% of actual fare)
SIGMOID_SCALING_FACTOR_RATIO = 0.1  #steepness of acceptance curve

# DQN Agent Hyperparameters
LEARNING_RATE = 0.001  #optimizer learning rate
GAMMA = 0.95  #discount factor for future rewards (Q-learning)
EPSILON_INITIAL = 1.0  #initial exploration rate
EPSILON_MIN = 0.01  #minimum exploration rate after decay
EPSILON_DECAY = 0.995  #decay multiplier per training batch
MEMORY_BUFFER_SIZE = 2000  #experience replay buffer size

# Training Configuration
NUM_TRAINING_EPISODES = 100  #total training episodes
BATCH_SIZE = 32  #minibatch size for experience replay
TARGET_UPDATE_FREQUENCY = 10  #update target network every N episodes

# Testing Configuration
NUM_TEST_EPISODES = 20  #number of test episodes
STEPS_PER_TEST_EPISODE = 100  #steps per test episode

