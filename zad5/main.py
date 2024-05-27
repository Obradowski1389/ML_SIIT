import sys
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import v_measure_score
from sklearn.mixture import GaussianMixture

numerical_feature = ['Year','Population','GDP per Capita','Urban Population','Life Expectancy','Surface Area','Literacy Rate']

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

    gmm = GaussianMixture(n_components=len(clusters), random_state=42)

    gmm.fit(X)

    y_test = test_data['region']
    X_test = test_data.drop(['region','Year'], axis=1)

    predictions = gmm.predict(X_test)

    v_measure = v_measure_score(y_test, predictions)

    print(v_measure)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])