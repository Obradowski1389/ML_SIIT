import numpy as np
import pandas as pd
from scipy import stats
import time
class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        variance = np.var(y)

        if (self.max_depth is not None and depth >= self.max_depth) or \
                (self.min_samples_split is not None and n_samples < self.min_samples_split) or \
                (np.unique(y).size == 1):
            return {"value": np.mean(y)}

        best_variance_reduction = 0
        best_feature_idx = None
        best_threshold = None
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_idx] <= threshold)[0]
                right_indices = np.where(X[:, feature_idx] > threshold)[0]

                if left_indices.size < self.min_samples_leaf or right_indices.size < self.min_samples_leaf:
                    continue

                left_variance = np.var(y[left_indices])
                right_variance = np.var(y[right_indices])
                weighted_variance = (left_variance * len(left_indices) + right_variance * len(right_indices)) / n_samples
                variance_reduction = variance - weighted_variance

                if variance_reduction > best_variance_reduction:
                    best_variance_reduction = variance_reduction
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        if best_variance_reduction > 0:
            left_subtree = self._build_tree(X[best_left_indices], y[best_left_indices], depth + 1)
            right_subtree = self._build_tree(X[best_right_indices], y[best_right_indices], depth + 1)
            return {"feature_idx": best_feature_idx, "threshold": best_threshold,
                    "left": left_subtree, "right": right_subtree}
        else:
            return {"value": np.mean(y)}

    def predict(self, X):
        X = np.array(X)  # Convert X to NumPy array
        return np.array([self._predict_tree(x, self.tree_) for x in X])

    def _predict_tree(self, x, tree):
        if "value" in tree:
            return tree["value"]
        feature_value = x[tree["feature_idx"]]
        if feature_value <= tree["threshold"]:
            return self._predict_tree(x, tree["left"])
        else:
            return self._predict_tree(x, tree["right"])

class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = []

    def fit(self, X, y):
        self.tree_ = []
        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split,
                                          min_samples_leaf=self.min_samples_leaf)
            tree.fit(X_bootstrap, y_bootstrap)
            self.tree_.append(tree)

    def predict(self, X):
        X = np.array(X)  # Convert X to NumPy array
        predictions = np.array([tree.predict(X) for tree in self.tree_])
        return np.mean(predictions, axis=0)




def calculate_rmse(y_true, y_pred):
    residuals = y_true - y_pred
    squared_residuals = residuals ** 2
    mean_squared_residuals = squared_residuals.mean()
    rmse = np.sqrt(mean_squared_residuals)
    return rmse


def main2(k = 41600):
    
    # df = pd.read_csv("data/temp.tsv", sep='\t')
    df_train = pd.read_csv("data/train.tsv", sep='\t')
    df_test = pd.read_csv("data/test.tsv", sep='\t')

    df_train.drop_duplicates(inplace=True)
    df_train.reset_index(drop=True, inplace=True)

    df_train.drop(columns=['Grad'], inplace=True)
    df_test.drop(columns=['Grad'], inplace=True)

    categorical_columns = df_train.select_dtypes(include=object).columns.tolist()
    numerical_columns = df_train.select_dtypes(exclude=object).columns.tolist()
    numerical_columns.remove('Cena')
    for column in categorical_columns:
        df_train[column] = pd.factorize(df_train[column])[0]
        df_test[column] = pd.factorize(df_test[column])[0]

    for column in numerical_columns:
        mean = df_train[column].mean()
        std = df_train[column].std()
        df_train[column] = (df_train[column] - mean) / std
        df_test[column] = (df_test[column] - mean) / std

    y_train = df_train['Cena']
    X_train = df_train.drop(columns=['Cena'])
    y_test = df_test['Cena']
    X_test = df_test.drop(columns=['Cena'])

    print(df_train.shape)
    print(df_train.columns)
    print(df_train.head())

    X_train = X_train.values
    y_train = y_train.values
    start_time = time.time()

# Code you want to time goes here...

    model = RandomForestRegressor(n_estimators=364, max_depth=46, min_samples_split=2, min_samples_leaf=2)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    rmse = calculate_rmse(y_test, y_pred)
    print(rmse)

    end_time = time.time()
    execution_time = end_time - start_time 

    print(f"Execution time: {execution_time} seconds") 


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
# import pandas as pd
# import numpy as np

# def main(k=41600):
#     def objective(trial):
#         n_estimators = trial.suggest_int('n_estimators', 100, 1000)
#         max_depth = trial.suggest_int('max_depth', 10, 50)
#         min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
#         min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        
#         model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
#                                     min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
        
#         score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        
#         return score.mean()

#     df = pd.read_csv("data/temp.tsv", sep='\t')

#     # Pogledaj
#     df.head()
#     df.tail()
#     df.info()

#     df.drop_duplicates(inplace=True)
#     df = df[df['Cena'] <= k]

#     y = df['Cena']
#     df.drop(columns=['Grad', 'Cena'], inplace=True)

#     categorical_columns = df.select_dtypes(include=object).columns.tolist()
#     numerical_columns = df.select_dtypes(exclude=object).columns.tolist()

#     label_encoder = LabelEncoder()
#     for column in categorical_columns:
#         df[column] = label_encoder.fit_transform(df[column])

#     scaler = StandardScaler()
#     df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


#     print(categorical_columns)
#     print(numerical_columns)
#     X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

#     # study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=42))
#     # study.optimize(objective, n_trials=250)
#     # print(study.best_params)
#     # best_params = study.best_params
#     # optuna.visualization.plot_optimization_history(study)
#     # optuna.visualization.plot_slice(study, params=['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'])

#     best_model = RandomForestRegressor(n_estimators=364, max_depth=46, 
#                                     min_samples_split=2, min_samples_leaf=2)
#     best_model.fit(X_train, y_train)

#     print(best_model.score(X_train, y_train))

#     y_pred = best_model.predict(X_test)
#     rmse = calculate_rmse(y_test, y_pred)
#     print(f'Mean Squared Error: {rmse}')
#     return rmse


# if __name__ == '__main__':
#     # temp = {}
#     # for i in range(40000, 45000, 200):
#     mse = main(41600)
#     print(mse)
#     # temp[i] = mse
        
#     # sorted_data = sorted(temp.items(), key=lambda x: x[1])
#     # min_key = sorted_data[0][0]

#     # print("Min key:", min_key, "\tRMSE:", temp[min_key])