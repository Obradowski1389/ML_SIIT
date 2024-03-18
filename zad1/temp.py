import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def linear_regression_fit(X_train, y_train):
    X_train = np.column_stack((np.ones(len(X_train)), X_train))
    coefficients = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    
    return coefficients


# Z-score
def normalize(X):
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    X_normalized = (X - mean) / std_dev
    return X_normalized, mean, std_dev


def linear_regression_predict(X, coefficients):
    X = np.column_stack((np.ones(len(X)), X))
    predictions = X @ coefficients
    
    return predictions


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true )**2))


def plot_data(X, Y):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X, Y)
    plt.show()

def plot_data_with_regression_line(X, y, coefficients):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X, y, label='Data')
    
    # Predict Y values using the linear regression model
    X_regression = np.linspace(np.min(X), np.max(X), 100)
    y_regression = linear_regression_predict(X_regression, coefficients)
    
    # Plot the regression line
    ax.plot(X_regression, y_regression, color='red', label='Linear Regression')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.show()


def main():
    data = pd.read_csv('data/train.csv')
    X = data['X']
    y = data['Y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # X_train_normalized, mean_train, std_dev_train = normalize(X_train)
    # X_test_normalized = (X_test - mean_train) / std_dev_train   # Normalize test data

    coefficients = linear_regression_fit(X_train, y_train)

    train_predictions = linear_regression_predict(X_train, coefficients)
    test_predictions = linear_regression_predict(X_test, coefficients)

    train_rmse = root_mean_squared_error(y_train, train_predictions)
    test_rmse = root_mean_squared_error(y_test, test_predictions)

    print("Train Predictions:", train_rmse)
    print("Test Predictions:", test_rmse)

    plot_data_with_regression_line(X_train, y_train, coefficients)


if __name__ == "__main__":
    main()