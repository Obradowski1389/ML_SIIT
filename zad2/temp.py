import pandas as pd
import numpy as np
import sys

df = None


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
        sum_squared_diff = 0.0

        for i in range(len(x)):
            sum_squared_diff += (x[i] - y[i]) ** 2

        return sum_squared_diff ** 0.5
    

def encode(train, test):
    global df

    nominal = {
        'Karoserija': df['Karoserija'].unique(),
        'Gorivo': df['Gorivo'].unique(),
        'Marka': df['Marka'].unique(),
    }

    # label encoding for binary
    train['Menjac'] = train['Menjac'].map({'Manuelni': 0, 'Automatski': 1})
    test['Menjac'] = test['Menjac'].map({'Manuelni': 0, 'Automatski': 1})
    # one hot encoding for nominal
    for col in nominal:
        for category in nominal[col]:
            train[category] = (train[col] == category).astype(int)
            test[category] = (test[col] == category).astype(int)
        train.drop(col, axis=1, inplace=True)
        test.drop(col, axis=1, inplace=True)

    return train, test


def normalize(train, test, numerical_columns):
    for col in numerical_columns:
        if col == 'Cena':
            continue
        mean = train[col].mean()
        std = train[col].std()
        train[col] = (train[col] - mean) / std
        test[col] = (test[col] - mean) / std
    return train, test


def calculate_rmse(y_true, y_pred):
    residuals = y_true - y_pred
    squared_residuals = residuals ** 2
    mean_squared_residuals = squared_residuals.mean()
    rmse = np.sqrt(mean_squared_residuals)
    return rmse


def main(train_path, test_path):
    global df
    train = pd.read_csv(train_path, sep='\t')
    test = pd.read_csv(test_path, sep='\t')

    df = pd.concat([train, test])

    train = train[train['Godina proizvodnje'] >= 1900]
    train = train[train['Cena'] <= 63000]
    train.drop_duplicates(inplace=True)
    train.reset_index(drop=True, inplace=True)

    numerical_columns = df.select_dtypes(exclude=object).columns.tolist()
    numerical_columns.remove('Cena')

    train, test = normalize(train, test, numerical_columns)

    train, test = encode(train, test)

    y_train = train['Cena']
    X_train = train.drop(columns= ['Cena', 'Grad'], axis=1)
    y_test = test['Cena']
    X_test = test.drop(columns= ['Cena', 'Grad'], axis=1)
    
    knn = KNNRegressor(k=10, dist=KNNRegressor.euclidean_distance)
    knn.fit(X_train.values, y_train.values)
    y_pred = knn.predict(X_test.values)
    return calculate_rmse(y_test, y_pred)

    


if __name__ == "__main__":
    # main("data/train.tsv", "data/test.tsv")

    # min_rmse = math.inf
    # best_k = 0
    # best_price = 0
    #cena
    # for i in range(40000, 70000, 1000):
    #     #k
    #     for j in range(1, 30):
    rmse = main(sys.argv[1], sys.argv[2])
            # print(rmse)
            # if rmse < min_rmse:
            #     min_rmse = rmse
            #     best_k = j
            #     best_price = i
    # print(f'Min RMSE: {min_rmse}, best k: {best_k}, best price: {best_price}')
    print(rmse)

