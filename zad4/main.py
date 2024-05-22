import sys
import pandas as pd
from sklearn.impute import KNNImputer
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


    y = train_data_final['Genre']
    X = train_data_final.drop('Genre', axis=1)

    # ansambl
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=20,
        min_samples_split=7,
        random_state=42
    )
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.3,
        random_state=42
    )

    voting_clf = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
    voting_clf.fit(X, y)


    y_test = test_data['Genre']

    # Zakomentarisati liniju ispod kada se deploy-uje model na platformi
    # test_data[numerical_features] = imputer.transform(test_data[numerical_features])
    platform_encoded_test = encoder.transform(test_data[categorical_features])
    platform_encoded_test_df = pd.DataFrame(platform_encoded_test.toarray(), columns=encoder.get_feature_names_out(categorical_features))

    test_data = pd.concat([test_data[numerical_features], platform_encoded_test_df], axis=1)

    test_predictions = voting_clf.predict(test_data)

    macro_f1_test = f1_score(y_test, test_predictions, average='macro')
    print(macro_f1_test)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])