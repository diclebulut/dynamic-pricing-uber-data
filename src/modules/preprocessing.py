import pandas as pd
from modules.config import DATA_PATH, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv(DATA_PATH)
def simple_preprocessing():              
    categorical_features = CATEGORICAL_FEATURES
    numerical_features = NUMERICAL_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].values
    return preprocessor, X, y, categorical_features, numerical_features