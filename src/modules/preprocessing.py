import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
from modules.config import VERSION, DATA_PATH, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN, WEATHER_NYC_PATH
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def simple_preprocessing(df,categorical_features, numerical_features):              
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        sparse_threshold=0  # Force dense output
    )
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].values
    return preprocessor, X, y, categorical_features, numerical_features

def preprocess_for_v2():
    trips = pq.read_table(DATA_PATH)  
    trips = trips.to_pandas()
    weather = pd.read_csv(WEATHER_NYC_PATH)
    uber_trips = trips[trips["hvfhs_license_num"] == "HV0003"]
    uber_trips = uber_trips[uber_trips['trip_miles'] > 0]

    uber_trips['out_of_base_dispatch_flag'] = np.where(uber_trips['dispatching_base_num'] != uber_trips['originating_base_num'], 1, 0)
    uber_trips['airport_fee'] = uber_trips['airport_fee'].fillna(0)
    uber_trips['final_fare'] = uber_trips["base_passenger_fare"] + uber_trips["tolls"] + uber_trips['bcf'] + uber_trips['sales_tax'] + uber_trips['congestion_surcharge'] + uber_trips["airport_fee"]
    uber_trips['uber_profit'] = uber_trips['final_fare'] - uber_trips['driver_pay']
    uber_trips['driver_final_pay'] = uber_trips['driver_pay'] + uber_trips['tips'] - (uber_trips["tolls"] + uber_trips['bcf'] + uber_trips['sales_tax'] + uber_trips['congestion_surcharge'] + uber_trips["airport_fee"])
    uber_trips['uber_profit_per_mile'] = uber_trips['uber_profit'] / uber_trips['trip_miles']
    uber_trips['uber_profit_per_second'] = uber_trips['uber_profit'] / uber_trips['trip_time']
    uber_trips["pickup_date"] = uber_trips["pickup_datetime"].dt.date
    uber_trips['wait'] = round((uber_trips['pickup_datetime'] - uber_trips['request_datetime']).dt.total_seconds(), 0)
    uber_trips["day_of_month"] = pd.to_datetime(uber_trips["pickup_date"]).dt.day

    hours = uber_trips["pickup_datetime"].dt.hour
    conditions = [
        (hours >= 23) | (hours < 1),   
        (hours >= 1) & (hours < 5),    
        (hours >= 5) & (hours < 11),   
        (hours >= 11) & (hours < 17), 
        (hours >= 17) & (hours < 23),  
    ]
    labels = ["night", "late", "morning", "midday", "evening"]

    uber_trips["time_of_day"] = np.select(conditions, labels, default="night")
    uber_trips["pickup_date"] = pd.to_datetime(uber_trips["pickup_datetime"]).dt.normalize()
    weather["datetime"] = pd.to_datetime(weather["datetime"]).dt.normalize()

    uber_trips = uber_trips.merge(
        weather[["datetime", "precip", "preciptype"]],
        left_on="pickup_date",
        right_on="datetime",
        how="left"
    )

    uber_trips['preciptype'] = uber_trips['preciptype'].fillna('None')
    uber_trips.drop(columns=["hvfhs_license_num", "dispatching_base_num", "originating_base_num",
                            "on_scene_datetime", "access_a_ride_flag", "wav_request_flag",
                            "wav_match_flag", "base_passenger_fare", "tolls", "bcf", "sales_tax", 
                            "congestion_surcharge", "airport_fee", "driver_pay", "pickup_datetime",
                            "request_datetime", "dropoff_datetime", "datetime"], inplace=True)
    uber_trips = uber_trips.reset_index(drop=True)

    uber_trips = uber_trips[uber_trips['day_of_month'] < 2]
    uber_trips = uber_trips.head(1000)
    #uber_trips["ride_id"] = uber_trips.index + 1
    return uber_trips


def preprocess():
    data_path = Path(DATA_PATH)

    if data_path.suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH, compression="infer", encoding="utf-8")

    if VERSION == "V1":
        preprocessor, X, y, categorical_features, numerical_features = simple_preprocessing(df, CATEGORICAL_FEATURES, NUMERICAL_FEATURES)
    elif VERSION == "V2":
        df = preprocess_for_v2()
        preprocessor, X, y, categorical_features, numerical_features = simple_preprocessing(df, CATEGORICAL_FEATURES, NUMERICAL_FEATURES)
    else:
        raise ValueError("Unsupported VERSION specified in config.py")
    return preprocessor, X, y, categorical_features, numerical_features









