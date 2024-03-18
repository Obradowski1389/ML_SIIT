import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def train_test_split(X, y, shuffle, test_size=0.2):
    num_samples = len(X)
    num_test_samples = int(test_size * num_samples)

    if shuffle:
        indices = np.random.permutation(num_samples)
        test_indices = indices[:num_test_samples]
        train_indices = indices[num_test_samples:]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
    else:
        X_train, X_test = X[:-num_test_samples], X[-num_test_samples:]
        y_train, y_test = y[:-num_test_samples], y[-num_test_samples:]
    return X_train, X_test, y_train, y_test


def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df >= lower_bound) & (df <= upper_bound)]


# Z-score
def normalize(X):
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    X_normalized = (X - mean) / std_dev
    return X_normalized, mean, std_dev


def compute_cost(X, y, coefficients):
    m = len(y)
    predictions = X @ coefficients
    cost = (1 / (2 * m)) * np.sum((predictions - y)**2)
    return cost


# Batch
def gradient_descent(X, y, coefficients, learning_rate, num_iterations):
    m = len(y)
    costs = []
    for _ in range(num_iterations):
        predictions = X @ coefficients
        error = predictions - y
        gradient = (1 / m) * (X.T @ error)
        coefficients -= learning_rate * gradient
        cost = compute_cost(X, y, coefficients)
        costs.append(cost)
    return coefficients, costs


def plot_data(X, Y):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X, Y)
    plt.show()

def main():
    data = pd.read_csv('data/train.csv')

    X = data[['X']].values
    y = data['Y'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)

    X_train_normalized, mean_train, std_dev_train = normalize(X_train)
    X_test_normalized = (X_test - mean_train) / std_dev_train   # Normalize test data

    initial_coefficients = np.zeros((X_train_normalized.shape[1] + 1, 1))
    learning_rate = 0.001
    num_iterations = 100000
    X_train_with_intercept = np.hstack((np.ones((X_train_normalized.shape[0], 1)), X_train_normalized))
    trained_coefficients, costs = gradient_descent(X_train_with_intercept, y_train.reshape(-1, 1), initial_coefficients, learning_rate, num_iterations)

    X_test_with_intercept = np.hstack((np.ones((X_test_normalized.shape[0], 1)), X_test_normalized))
    y_train_pred = X_train_with_intercept @ trained_coefficients
    y_test_pred = X_test_with_intercept @ trained_coefficients

    # Denormalize
    y_train_pred_denormalized = (y_train_pred * std_dev_train) + mean_train
    y_test_pred_denormalized = (y_test_pred * std_dev_train) + mean_train

    # RMSE
    train_rmse = np.sqrt(np.mean((y_train - y_train_pred_denormalized)**2))
    test_rmse = np.sqrt(np.mean((y_test - y_test_pred_denormalized)**2))


    print("Training RMSE:", train_rmse)
    print("Testing RMSE:", test_rmse)


if __name__ == '__main__':
    main()