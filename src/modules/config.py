# ============================================
# Data Version
# ============================================

VERSION = 'V2' 
# ============================================
# Data Paths
# ============================================

DATA_PATH_V1 = 'data/uber_data.csv'
DATA_PATH_V2 = 'data/fhvhv_tripdata_2021-01.parquet'
TAXI_ZONE_PATH = 'data/taxi_zone_lookup.csv'
WEATHER_NYC_PATH = 'data/nyc 2021-01-01 to 2021-12-31.csv'

# ============================================
# Columns as features
# ============================================

CATEGORICAL_FEATURES_V1 = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']
NUMERICAL_FEATURES_V1 = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 'Expected_Ride_Duration']
TARGET_COLUMN_V1 = 'Historical_Cost_of_Ride'

CATEGORICAL_FEATURES_V2 = ['PULocationID', 'DOLocationID', 'shared_request_flag', 
                        'shared_match_flag', 'out_of_base_dispatch_flag',
                        'time_of_day','preciptype'
                        ]
NUMERICAL_FEATURES_V2 = ['trip_miles', 'trip_time', 'tips',
                       'wait', 'precip', 'day_of_month']
TARGET_COLUMN_V2 = 'final_fare'

# ============================================
# Active (versioned) config
# ============================================

if VERSION == 'V1':
    DATA_PATH = DATA_PATH_V1
    CATEGORICAL_FEATURES = CATEGORICAL_FEATURES_V1
    NUMERICAL_FEATURES = NUMERICAL_FEATURES_V1
    TARGET_COLUMN = TARGET_COLUMN_V1
elif VERSION == 'V2':
    DATA_PATH = DATA_PATH_V2
    CATEGORICAL_FEATURES = CATEGORICAL_FEATURES_V2
    NUMERICAL_FEATURES = NUMERICAL_FEATURES_V2
    TARGET_COLUMN = TARGET_COLUMN_V2
else:
    raise ValueError(f"Unsupported VERSION: {VERSION}")

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