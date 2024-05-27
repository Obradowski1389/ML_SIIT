import sys
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import v_measure_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, train_test_split

numerical_feature = ['Population','GDP per Capita','Urban Population','Life Expectancy','Surface Area','Literacy Rate']

def load_data(train, test):
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)

    return train_data, test_data

def main(train, test):
    train_data, test_data = load_data(train, test)

    imputer = KNNImputer()
    train_data[numerical_feature] = imputer.fit_transform(train_data[numerical_feature])
    
    # TODO: delete
    test_data[numerical_feature] = imputer.transform(test_data[numerical_feature])
    
    clusters = pd.concat([train_data['region'], test_data['region']]).unique()

    y = train_data['region']
    X = train_data.drop(['region','Year'], axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    gmm = GaussianMixture(n_components=len(clusters))

    param_grid = {
        'random_state': [0, 1, 2, 3, 7, 20, 42],
        'init_params': ['random', 'kmeans', 'random_from_data', 'k-means++'],
        'covariance_type': ['full', 'tied', 'diag', 'spherical'],
        'max_iter': [100, 200, 300, 400, 500],
    }

    grid_search = GridSearchCV(gmm, param_grid, cv=5, scoring='v_measure_score', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f'Best Hyperparameters: {best_params}')
    print(f'Best V Measure Score on Validation Set: {best_score}')

    y_val_pred = best_model.predict(X_val)
    v_measure = v_measure_score(y_val, y_val_pred)
    print(f'V Measure Score on Validation Set: {v_measure}')


    # gmm.fit(X)

    y_test = test_data['region']
    X_test = test_data.drop(['region','Year'], axis=1)

    predictions = best_model.predict(X_test)

    v_measure = v_measure_score(y_test, predictions)
    print(f'V Measure Score on Test Set: {v_measure}')
    # print(v_measure)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])