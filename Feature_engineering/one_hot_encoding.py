from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd


def one_hot_encode_features(df_train, df_test, categorical_cols):
    df, dft = df_train.copy(), df_test.copy()
    # num_of_values_dict = {}
    for column in categorical_cols:
        # Label encoding part
        col_le = LabelEncoder()
        col_labels = col_le.fit(df[column].astype(str))
        col_labels_train = col_le.transform(df[column].astype(str))
        col_labels_test = col_le.transform(dft[column].astype(str))
        df['{}_label'.format(column)] = col_labels_train
        dft['{}_label'.format(column)] = col_labels_test

        # OHE part
        col_ohe = OneHotEncoder()
        col_feature_arr = col_ohe.fit(df[['{}_label'.format(column)]])
        col_feature_arr_train = col_ohe.transform(df[['{}_label'.format(column)]]).toarray()
        col_feature_arr_test = col_ohe.transform(dft[['{}_label'.format(column)]]).toarray()
        col_feature_labels = [column + "_" + x for x in col_le.classes_]
        # num_of_values_dict[column] = len(col_feature_labels)
        col_features_train = pd.DataFrame(col_feature_arr_train, columns=col_feature_labels)
        col_features_test = pd.DataFrame(col_feature_arr_test, columns=col_feature_labels)

        # Dropping encoded columns
        df = pd.concat([df, col_features_train], axis=1)
        df = df.drop(["{}_label".format(column), column], axis=1)
        dft = pd.concat([dft, col_features_test], axis=1)
        dft = dft.drop(["{}_label".format(column), column], axis=1)
    return df, dft
