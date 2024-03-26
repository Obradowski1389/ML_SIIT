import pandas as pd
import numpy as np
import sys

def load_data(path):
    data = pd.read_csv(path)
    return data

def predict(X, theta):
    return np.dot(X, theta)

def compute_cost(X, Y, theta):
    m = len(Y)
    predictions = predict(X, theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - Y))
    return cost

def gradient_descent(X, Y, theta, learning_rate, iterations):
    m = len(Y)
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        predictions = predict(X, theta)
        theta = theta - (1/m) * learning_rate * (X.T.dot(predictions - Y))
        cost_history[i] = compute_cost(X, Y, theta)
    return theta, cost_history

def rmse(Y, Y_pred):
    return np.sqrt(np.mean((Y_pred - Y)**2))

def min_max_normalization(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def z_score_normalization(data):
    return (data - np.mean(data)) / np.std(data)

def train_test_split(X, y, train_size=0.8):
    num_samples = len(X)
    train_size = int(num_samples * train_size)
    indices = np.random.permutation(num_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test, test_indices

def main(path):
    data = load_data(path)
    X = data["X"].values.reshape(-1, 1)
    y = data["Y"].values

    # Apply normalization
    X_normalized_minmax = min_max_normalization(X)
    y_normalized_minmax = min_max_normalization(y)
    X_normalized_zscore = z_score_normalization(X)
    y_normalized_zscore = z_score_normalization(y)

    # Split the normalized data
    X_train_minmax, X_test_minmax, Y_train_minmax, Y_test_minmax, test_indices_minmax = train_test_split(X_normalized_minmax, y_normalized_minmax)
    X_train_zscore, X_test_zscore, Y_train_zscore, Y_test_zscore, test_indices_zscore = train_test_split(X_normalized_zscore, y_normalized_zscore)

    theta_minmax = np.zeros(X_train_minmax.shape[1])
    theta_zscore = np.zeros(X_train_zscore.shape[1])

    # Set hyperparameters
    learning_rate = 0.01
    iterations = 1000

    # Train the model with min-max normalized data
    theta_minmax, _ = gradient_descent(X_train_minmax, Y_train_minmax, theta_minmax, learning_rate, iterations)

    # Train the model with z-score normalized data
    theta_zscore, _ = gradient_descent(X_train_zscore, Y_train_zscore, theta_zscore, learning_rate, iterations)

    # Make predictions on the test set with min-max normalized data
    Y_pred_minmax = predict(X_test_minmax, theta_minmax)

    # Make predictions on the test set with z-score normalized data
    Y_pred_zscore = predict(X_test_zscore, theta_zscore)

    # Denormalize the predictions
    Y_pred_minmax_denormalized = Y_pred_minmax * (np.max(y) - np.min(y)) + np.min(y)
    Y_pred_zscore_denormalized = Y_pred_zscore * np.std(y) + np.mean(y)

    # Calculate RMSE for min-max normalized predictions
    rmse_minmax = rmse(data["Y"].values[test_indices_minmax], Y_pred_minmax_denormalized)
    print("RMSE for Min-Max Normalized Predictions:", rmse_minmax)

    # Calculate RMSE for z-score normalized predictions
    rmse_zscore = rmse(data["Y"].values[test_indices_zscore], Y_pred_zscore_denormalized)
    print("RMSE for Z-Score Normalized Predictions:", rmse_zscore)

if __name__ == "__main__":
    path = sys.argv[1]
    main(path)
