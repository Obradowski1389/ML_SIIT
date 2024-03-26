import pandas as pd
import numpy as np
import scipy
import sys
import seaborn as sns
import matplotlib.pyplot as plt

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
        categories = pd.concat([train_data[column], test_data[column]]).unique()
        label_map = {category: i for i, category in enumerate(categories)}
        label_mappings[column] = label_map
        train_data[column] = train_data[column].map(label_map)
        test_data[column] = test_data[column].map(label_map)
    
    return train_data, test_data, 
    

def calculate_rmse(y_true, y_pred):
    residuals = y_true - y_pred
    squared_residuals = residuals ** 2
    mean_squared_residuals = squared_residuals.mean()
    rmse = np.sqrt(mean_squared_residuals)
    return rmse


def main(train_path, test_path):
    train_data = pd.read_csv(train_path, sep='\t')
    test_data = pd.read_csv(test_path, sep='\t')
    train_data, test_data = label_encode(train_data, test_data)
    
    correlation_matrix = train_data.corr()  
    # visualize_correlation_matrix(correlation_matrix)

    # TODO drop more columns based of correlation matrix
    y_train = train_data['Cena']
    X_train = train_data.drop(columns=['Cena'])
    y_test = test_data['Cena']
    X_test = test_data.drop(columns=['Cena'])



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
