import pandas as pd
from modules.config import DATA_PATH
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv(DATA_PATH)
def simple_preprocessing():              
    categorical_features = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']
    numerical_features = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 'Expected_Ride_Duration']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    X = df.drop(columns=['Historical_Cost_of_Ride'])
    y = df['Historical_Cost_of_Ride'].values
    return preprocessor, X, y, categorical_features, numerical_features