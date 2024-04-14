import numpy as np
import pandas as pd
import operator
from functools import reduce
from math import sqrt
import matplotlib.pyplot as plt


class KNNRegressor:
    def __init__(self, k, dist):
        self.k = k
        self.distance = dist
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X_test):
        predictions = []
        for sample in X_test:
            distances = np.array([self.distance(sample, x) for x in self.X_train])
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = [self.y_train[i] for i in nearest_indices] 
            prediction = np.mean(nearest_labels)
            predictions.append(prediction)
        return np.array(predictions)
    
    @staticmethod
    def euclidean_distance(x, y):
        return sqrt(reduce(operator.add, map(lambda a, b: (a - b) ** 2, x, y)))
    

def label_encode(train_data, test_data):
    label_mappings = {}
    
    categorical_columns = ['Marka', 'Grad']
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

def feature_normalization(data, binary, nominal, range_cols=None, norm_params=[], feature_to_predict='Cena'):
    if range_cols is None:
        range_cols = [col for col in data.columns.values if col not in list(binary.keys()) + list(nominal.keys()) + [feature_to_predict]]

    rp_initialized = len(norm_params) != 0

    # Normalization of features whose values are in some range
    for i in range(len(range_cols)):
        col_name = range_cols[i]
        col_values = data[col_name]

        if not rp_initialized:
            median = np.median(col_values)
            std = np.std(col_values)
            norm_params.append((median, std))
        else:
            median = norm_params[i][0]
            std = norm_params[i][1]

        data[col_name] = (col_values - median) / std

    # Set binary feature values from string to 0 or 1
    for col_name, bin_values in binary.items():
        data[col_name] = data[col_name].map({bin_values[0]: 1, bin_values[1]: 0})

    # Create new feature for every category from nominal feature
    for col_name in nominal:
        for cat in nominal[col_name]:
            data[cat] = list(map(lambda x: int(x == cat), data[col_name]))

        data.drop(col_name, axis=1, inplace=True)

    # Set column to be predicted at the last position
    y_values = data[feature_to_predict]
    data.drop(feature_to_predict, axis=1, inplace=True)
    data[feature_to_predict] = y_values

    return data


def plot_expected_vs_predicted(y_expected, y_predicted):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_expected, y_predicted, color='blue', alpha=0.5)
    plt.plot([min(y_expected), max(y_expected)], [min(y_expected), max(y_expected)], color='red', linestyle='--')
    plt.title('Expected vs Predicted Prices')
    plt.xlabel('Expected Price')
    plt.ylabel('Predicted Price')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main(train_path, test_path, k):
    selected_features = ["Marka", "Grad", "Cena", "Godina proizvodnje", "Karoserija", "Gorivo", "Zapremina motora", "Kilometraza", "Konjske snage", "Menjac"]
    train_data = pd.read_csv(train_path, sep='\t')
    test_data = pd.read_csv(test_path, sep='\t')

    train_data, test_data = label_encode(train_data, test_data) # Samo Marka i Grad

    binary_features = {
        'Menjac': ['Manuelni', 'Automatski'],
    }

    nominal_features = {
        'Karoserija': ["Hečbek", "Limuzina", "Karavan", "Džip/SUV", "Monovolumen (MiniVan)", "Kupe", "Kabriolet/Roadster", "Pickup"],
        'Gorivo': ["Dizel", "Benzin", "Benzin + Gas (TNG)", "Benzin + Metan (CNG)", "Hibridni pogon", "Hibridni pogon (Dizel)", "Hibridni pogon (Benzin)"],
    }

    
    norm_params = []
    train_data = feature_normalization(train_data, binary=binary_features, nominal=nominal_features, norm_params=norm_params)
    test_data = feature_normalization(test_data, binary=binary_features, nominal=nominal_features, norm_params=norm_params)

    y_train = train_data['Cena']
    X_train = train_data
    # X_train = train_data.drop(columns=['Grad', 'Gorivo'])
    y_test = test_data['Cena']
    X_test = test_data
    # X_test = test_data.drop(columns=['Grad', 'Gorivo'])

    knn = KNNRegressor(k=k, dist=KNNRegressor.euclidean_distance)

    knn.fit(X_train.values.tolist(), y_train.values.tolist())
    y_pred = knn.predict(X_test.values.tolist())
    
    plot_expected_vs_predicted(y_test, y_pred)
    rmse = calculate_rmse(y_test, y_pred)
    print("K: ", k, " : ", rmse)
    
    
if __name__ == '__main__':
    # main(sys.argv[1], sys.argv[2])
    # for i in range(1, 25):
    main("data/train.tsv", "data/test.tsv", 1)