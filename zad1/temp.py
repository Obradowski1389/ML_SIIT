import numpy as np
import pandas as pd
import sys


def train_test_split(X, y, test_size=0.2, shuffle=True):
    num_samples = len(X)
    num_test_samples = int(test_size * num_samples)
    
    if shuffle:
        indices = np.random.permutation(num_samples)
        test_indices = indices[:num_test_samples]
        train_indices = indices[num_test_samples:]
    else:
        test_indices = np.arange(num_samples - num_test_samples, num_samples)
        train_indices = np.arange(num_samples - num_test_samples)
    
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    
    return X_train, X_test, y_train, y_test


# Z-score
def normalize(X):
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    X_normalized = (X - mean) / std_dev
    return X_normalized, mean, std_dev


def linear_regression_fit(X_train, y_train):
    X_train = np.column_stack((np.ones(len(X_train)), X_train))
    coefficients = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    
    return coefficients


def linear_regression_predict(X, coefficients):
    X = np.column_stack((np.ones(len(X)), X))
    predictions = X @ coefficients
    
    return predictions


def calculate_rmse(y_true, y_pred):
    residuals = y_true - y_pred
    squared_residuals = residuals ** 2
    mean_squared_residuals = squared_residuals.mean()
    rmse = np.sqrt(mean_squared_residuals)
    return rmse


def main(path_train, path_test):
    data = pd.read_csv(path_train)
    X_train = data['X']
    y_train = data['Y']

    testing = pd.read_csv(path_test)
    X_test = testing['X']
    y_test = testing['Y']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    X_train_normalized, mean_train, std_dev_train = normalize(X_train)
    X_test_normalized = (X_test - mean_train) / std_dev_train   # Normalize test data

    coefficients = linear_regression_fit(X_train_normalized, y_train)

    # train_predictions = linear_regression_predict(X_train, coefficients)
    test_predictions = linear_regression_predict(X_test_normalized, coefficients)

    # train_rmse = calculate_rmse(y_train, train_predictions)
    test_rmse = calculate_rmse(y_test, test_predictions)

    print(test_rmse)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])