import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import f1_score


numerical_features = ['YoR', 'Sales_NA', 'Sales_EU', 'Sales_JP', 'Other_Sales']
categorical_features = ['Gaming_Platform']


def main(train, test):
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)

    imputer = KNNImputer()
    train_data[numerical_features] = imputer.fit_transform(train_data[numerical_features])

    encoder = OneHotEncoder()
    platform_encoded = encoder.fit_transform(train_data[categorical_features])
    platform_encoded_df = pd.DataFrame(platform_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_features))

    train_data_final = pd.concat([train_data[numerical_features], platform_encoded_df, train_data['Genre']], axis=1)

    # Podela na trening i validacioni skup
    y = train_data_final['Genre']
    X = train_data_final.drop('Genre', axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treniranje ansambl klasifikatora
    rf = RandomForestClassifier(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)

    voting_clf = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')

    # Podesavanje hiperparametara
    param_grid = {
        'rf__n_estimators': [100, 200],
        'gb__n_estimators': [100, 200]
    }
    grid_search = GridSearchCV(voting_clf, param_grid, cv=5, scoring='f1_macro')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Evaluacija modela
    y_val_pred = best_model.predict(X_val)
    macro_f1 = f1_score(y_val, y_val_pred, average='macro')
    print(f'Macro F1 Score on Validation Set: {macro_f1}')

    # Zakomentarisati kada se deploy-uje model na platformi
    test_data[numerical_features] = imputer.transform(test_data[numerical_features])
    platform_encoded_test = encoder.transform(test_data[categorical_features])
    platform_encoded_test_df = pd.DataFrame(platform_encoded_test.toarray(), columns=encoder.get_feature_names_out(categorical_features))

    test_data_final = pd.concat([test_data[numerical_features], platform_encoded_test_df], axis=1)

    # Predikcija na testnom skupu
    y_test = test_data['Genre']
    X_test = test_data_final

    test_predictions = best_model.predict(X_test)

    macro_f1_test = f1_score(y_test, test_predictions, average='macro')
    print(f'Macro F1 Score on Test Set: {macro_f1_test}')



if __name__ == '__main__':
    main('train.csv', 'test.csv')