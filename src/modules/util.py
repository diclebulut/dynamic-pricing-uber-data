import numpy as np
from sklearn.model_selection import train_test_split
from modules.config import TEST_SIZE, RANDOM_STATE

def intercept_reshaped_train_test(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    #bias term for intercept
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    #reshape
    X_train = X_train.T[np.newaxis, :]
    X_test = X_test.T[np.newaxis, :]
    y_train = y_train[np.newaxis, :]
    y_test = y_test[np.newaxis, :]
    return X_train, X_test, y_train, y_test