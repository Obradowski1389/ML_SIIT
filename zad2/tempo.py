from scipy import stats
import pandas as pd
import numpy as np


class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.trees)))
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        return np.mean(predictions, axis=1)

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        variance = np.var(y)

        if depth == self.max_depth or n_samples < self.min_samples_split or variance == 0:
            return np.mean(y)

        best_split = None
        best_gain = -np.inf
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            for threshold in np.unique(feature_values):
                left_indices = np.where(feature_values <= threshold)[0]
                right_indices = np.where(feature_values > threshold)[0]
                if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                    continue
                left_variance = np.var(y[left_indices])
                right_variance = np.var(y[right_indices])
                weighted_variance = (len(left_indices) / n_samples) * left_variance + (len(right_indices) / n_samples) * right_variance
                gain = variance - weighted_variance
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_idx, threshold)

        if best_gain == -np.inf:
            return np.mean(y)

        feature_idx, threshold = best_split
        left_indices = np.where(X[:, feature_idx] <= threshold)[0]
        right_indices = np.where(X[:, feature_idx] > threshold)[0]

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (feature_idx, threshold, left_subtree, right_subtree)

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree):
        if isinstance(tree, np.float64):
            return tree
        feature_idx, threshold, left_subtree, right_subtree = tree
        if x[feature_idx] <= threshold:
            return self._predict_tree(x, left_subtree)
        else:
            return self._predict_tree(x, right_subtree)


def calculate_rmse(y_true, y_pred):
    residuals = y_true - y_pred
    squared_residuals = residuals ** 2
    mean_squared_residuals = squared_residuals.mean()
    rmse = np.sqrt(mean_squared_residuals)
    return rmse


def main2(k = 41600):
    
    df = pd.read_csv("data/temp.tsv", sep='\t')

    df.drop_duplicates(inplace=True)
    df = df[df['Cena'] <= k]

    y = df['Cena']
    df.drop(columns=['Grad', 'Cena'], inplace=True)

    categorical_columns = df.select_dtypes(include=object).columns.tolist()
    numerical_columns = df.select_dtypes(exclude=object).columns.tolist()

    for column in categorical_columns:
        df[column] = pd.factorize(df[column])[0]

    df[numerical_columns] = stats.zscore(df[numerical_columns])

    for column in numerical_columns:
        mean = df[column].mean()
        std = df[column].std()
        df[column] = (df[column] - mean) / std

    indices = np.arange(len(df))

    train_size = int(0.8 * len(df))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train = df.iloc[train_indices]
    X_test = df.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    # print("X_train index:", X_train.index)
    # print("y_train index:", y_train.index)
    # print("X_train shape:", X_train.shape)
    # print("y_train shape:", y_train.shape)
    # print("Indices:", indices)
    print(X_train[677:678])
    print("Indices:", indices)
    print("Column names:", df.columns)
    model = RandomForestRegressor(n_estimators=364, max_depth=46, min_samples_split=2, min_samples_leaf=2)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    rmse = calculate_rmse(y_test, y_pred)
    print(rmse)

if __name__ == '__main__':
    main2()













# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import cross_val_score
# import seaborn as sns
# import matplotlib.pyplot as plt
# import optuna
# def main(k=41600):
    # def objective(trial):
    #     n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    #     max_depth = trial.suggest_int('max_depth', 10, 50)
    #     min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
    #     min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        
    #     model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
    #                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
        
    #     score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        
    #     return score.mean()

    # df = pd.read_csv("data/temp.tsv", sep='\t')

    # Pogledaj
    # df.head()
    # df.tail()
    # df.info()

    # df.drop_duplicates(inplace=True)
    # df = df[df['Cena'] <= k]

    # y = df['Cena']
    # df.drop(columns=['Grad', 'Cena'], inplace=True)

    # categorical_columns = df.select_dtypes(include=object).columns.tolist()
    # numerical_columns = df.select_dtypes(exclude=object).columns.tolist()

    # label_encoder = LabelEncoder()
    # for column in categorical_columns:
    #     df[column] = label_encoder.fit_transform(df[column])

    # scaler = StandardScaler()
    # df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # print(categorical_columns)
    # print(numerical_columns)
    # X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    # study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=42))
    # study.optimize(objective, n_trials=250)
    # print(study.best_params)
    # best_params = study.best_params
    # optuna.visualization.plot_optimization_history(study)
    # optuna.visualization.plot_slice(study, params=['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'])

    # best_model = RandomForestRegressor(n_estimators=364, max_depth=46, 
    #                                 min_samples_split=2, min_samples_leaf=2)
    # best_model.fit(X_train, y_train)

    # print(best_model.score(X_train, y_train))

    # y_pred = best_model.predict(X_test)
    # rmse = calculate_rmse(y_test, y_pred)
    # print(f'Mean Squared Error: {rmse}')
    # return rmse


# if __name__ == '__main__':
    # temp = {}
    # for i in range(40000, 45000, 200):
    #     mse = main(i)
    #     temp[i] = mse
        
    # sorted_data = sorted(temp.items(), key=lambda x: x[1])
    # min_key = sorted_data[0][0]

    # print("Min key:", min_key, "\tRMSE:", temp[min_key])