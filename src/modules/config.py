# ============================================
# Data Paths
# ============================================

DATA_PATH = 'data/fhvhv_tripdata_2021-01.parquet'
TAXI_ZONE_PATH = 'data/taxi_zone_lookup.csv'
WEATHER_NYC_PATH = 'data/nyc 2021-01-01 to 2021-12-31.csv'

# ============================================
# Columns as features
# ============================================

CATEGORICAL_FEATURES = ['PULocationID', 'DOLocationID', 'shared_request_flag', 
                        'shared_match_flag', 'out_of_base_dispatch_flag',
                        'time_of_day','preciptype'
                        ]
NUMERICAL_FEATURES = ['trip_miles', 'trip_time', 'tips',
                       'wait', 'precip', 'day_of_month']
TARGET_COLUMN = 'final_fare'

