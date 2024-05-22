import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV, train_test_split
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
    platform_encoded_df = pd.DataFrame(platform_encoded.toarray(),
                                       columns=encoder.get_feature_names_out(categorical_features))

    train_data_final = pd.concat([train_data[numerical_features], platform_encoded_df, train_data['Genre']], axis=1)

    # Podela na trening i validacioni skup
    y = train_data_final['Genre']
    X = train_data_final.drop('Genre', axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treniranje ansambl klasifikatora
    rf = RandomForestClassifier(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)

    voting_clf = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')

    # Podesavanje hiperparametara                           # 6 * 7 * 9 * 5 * 6 = 1890 kombinacija * 1.5min = 47.25h
    param_grid = {
        'rf__n_estimators': [50, 100, 150, 200, 250, 300],              # 6
        'rf__max_depth': [None, 5, 10, 15, 20, 25, 30],                 # 7
        'rf__min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],          # 9
        'gb__n_estimators': [100, 150, 200, 250, 300],                  # 5
        'gb__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]          # 6
    }
    grid_search = GridSearchCV(voting_clf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f'Best Hyperparameters: {best_params}')
    print(f'Best Macro F1 Score on Validation Set: {best_score}')
#     Best Hyperparameters: {'gb__learning_rate': 0.3, 'gb__n_estimators': 100, 'rf__max_depth': 20, 'rf__min_samples_split': 7, 'rf__n_estimators': 50}
# Best Macro F1 Score on Validation Set: 0.41240975569246824
# Macro F1 Score on Validation Set: 0.411779494966304
# Macro F1 Score on Test Set: 0.4262872369723919

    # Evaluacija modela na validacionom skupu
    y_val_pred = best_model.predict(X_val)
    macro_f1 = f1_score(y_val, y_val_pred, average='macro')
    print(f'Macro F1 Score on Validation Set: {macro_f1}')

    # Predikcija na testnom skupu
    test_data[numerical_features] = imputer.transform(test_data[numerical_features])
    platform_encoded_test = encoder.transform(test_data[categorical_features])
    platform_encoded_test_df = pd.DataFrame(platform_encoded_test.toarray(),
                                            columns=encoder.get_feature_names_out(categorical_features))

    test_data_final = pd.concat([test_data[numerical_features], platform_encoded_test_df], axis=1)

    # Predikcija na testnom skupu
    test_predictions = best_model.predict(test_data_final)

    # Macro F1 merenje na testnom skupu
    macro_f1_test = f1_score(test_data['Genre'], test_predictions, average='macro')
    print(f'Macro F1 Score on Test Set: {macro_f1_test}')


if __name__ == '__main__':
    main('train.csv', 'test.csv')
