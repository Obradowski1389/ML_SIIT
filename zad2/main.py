import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from sklearn import linear_model

def visualize_correlation_matrix(correlation_matrix):
    # Create a heatmap using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

def label_encode(train_data, test_data):
    label_mappings = {}
    
    categorical_columns = ['Marka', 'Grad', 'Karoserija', 'Gorivo', 'Menjac']
    for column in categorical_columns:
        categories = sorted(pd.concat([train_data[column], test_data[column]]).unique())
    
        label_map = {category: i for i, category in enumerate(categories)}
        label_mappings[column] = label_map
        
        train_data[column] = train_data[column].map(label_map)
        test_data[column] = test_data[column].map(label_map)
    
    return train_data, test_data



def calculate_rmse(y_true, y_pred):
    residuals = y_true - y_pred
    squared_residuals = residuals ** 2
    mean_squared_residuals = squared_residuals.mean()
    rmse = np.sqrt(mean_squared_residuals)
    return rmse

def predict(X, w, bias):
    return np.dot(X, w) + bias

def normalize_features(X_train, X_test):
    # z-score normalization
    mean = X_train.mean()
    std = X_train.std()
    X_train_normalized = (X_train - mean) / std
    X_test_normalized = (X_test - mean) / std
    return X_train_normalized, X_test_normalized

def fit_with_regularization(X, y, learning_rate=0.0001, max_iterations=10000, lambda_=0.01):
    weights = np.zeros(X.shape[1])
    bias = 0
    loss = []

    for _ in range(max_iterations):
        y_hat = np.dot(X, weights) + bias
        loss.append(calculate_rmse(y, y_hat))

        partial_w = (1 / X.shape[0]) * (2 * np.dot(X.T, (y_hat - y)))
        partial_b = (1 / X.shape[0]) * (2 * np.sum(y_hat - y))

        # L1 regularization
        regularization_term = lambda_ * weights
        partial_w += regularization_term

        weights -= learning_rate * partial_w
        bias -= learning_rate * partial_b

    return weights, bias, loss


def main(train_path, test_path):
    train_data = pd.read_csv(train_path, sep='\t')
    test_data = pd.read_csv(test_path, sep='\t')
    train_data, test_data = label_encode(train_data, test_data)

    train_data['Cena'] = np.log1p(train_data['Cena'])
    test_data['Cena'] = np.log1p(test_data['Cena'])
    
    y_train = train_data['Cena']
    X_train = train_data.drop(columns=['Cena', 'Grad', 'Gorivo'])
    y_test = test_data['Cena']
    X_test = test_data.drop(columns=['Cena', 'Grad', 'Gorivo'])

    # Feature scaling
    X_train_normalized, X_test_normalized = normalize_features(X_train, X_test)

    # Fit with regularization
    w, b, loss = fit_with_regularization(X_train_normalized, y_train, lambda_=1)

    # Prediction
    y_pred = predict(X_test_normalized, w, b)

    y_test = np.expm1(y_test)
    y_pred = np.expm1(y_pred)
    print(y_pred)
    rmse = calculate_rmse(y_test, y_pred)
    print(rmse)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
