import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt


def load_data(path):
    data = pd.read_csv(path)
    return data


def plot_data(X, Y):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X, Y)
    plt.show()

def initialize_theta(D): 
  return np.zeros([D, 1])

def lin_func(X, theta):
  assert X.ndim > 1
  assert theta.ndim > 1
  return np.dot(X, theta)

def batch_gradient(X, y, theta):
  return -2.0 * np.dot(X.T, (y - lin_func(X, theta)))

def update_function(theta, grads, step_size):
  return theta - step_size * grads

def GD_batch(X, y, learning_rate=0.01, num_iterations=1000):
    N, D = X.shape
    theta = initialize_theta(D)
    losses = []
    for _ in range(num_iterations): 
        ypred = lin_func(X, theta)
        loss = calculate_mse(y, ypred) 
        grads = batch_gradient(X, y, theta)
        theta = update_function(theta, grads, learning_rate)
        
        losses.append(loss)
    return losses


def GD_stochastic(X, y, learning_rate=0.01, num_iterations=1000):
    pass


def normal_equation(X, y):
    # pitala chatpgt - proveriti
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta


def train_model(X, y, method, learning_rate=0.01, num_iterations=1000):
    if method == 'GD_batch':
        return GD_batch(X, y, learning_rate, num_iterations)
    elif method == 'GD_stochastic':
        return GD_stochastic(X, y, learning_rate, num_iterations)
    elif method == 'normal_equation':
        return normal_equation(X, y)
    else:
        return None


def calculate_mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)


def normalize_data(data, method):   # Multiple types of normalization; add more?
    if method == 'min-max':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'z-score':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'log':
        # mislim da je ovo za outliere?!
        return np.log(data)
    else:
        return data


def handle_outliers(data, method): # Multiple types of handling outliers
    if method == 'remove':
        pass
    elif method == 'z-score':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return data[z_scores < 3]
    elif method == 'log':
        pass
    elif method == 'mean':
        pass
    elif method == 'robust-regression':
        pass
    else:
        return data


def main(path):
    data = load_data(path)
    print(data)
    X = data['X'].values.reshape(-1, 1)  
    Y = data['Y'].values

    plot_data(X, Y)
    
    optimization_algorithms = [GD_batch, GD_stochastic]
    normalization_methods = ['min-max', 'z-score', 'log']
    outlier_handling_methods = ['remove', 'z-score', 'log', 'mean', 'robust-regression']
    results = []

    for opt_algo in optimization_algorithms:
        for norm_method in normalization_methods:
            for outlier_method in outlier_handling_methods:
                
                normalized_X = normalize_data("", norm_method)
                cleaned_X = handle_outliers(normalized_X, outlier_method)

                predictions = train_model()
                mse = calculate_mse(predictions, Y)

                results.append({
                    'Optimization Algorithm': opt_algo.__name__,
                    'Normalization Method': norm_method,
                    'Outlier Handling Method': outlier_method,
                    'MSE': mse
                })

    
    results_df = pd.DataFrame(results)
    sorted_results_df = results_df.sort_values(by='MSE', ascending=True) 
    sorted_results_df.to_csv("results.csv", index=False)




if __name__ == "__main__":
    path = sys.argv[1]
    main(path)