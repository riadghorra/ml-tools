from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


def scale_continuous(df_train, df_val, df_test, continuous_cols, validation=True):
    df, dft = df_train.copy(), df_test.copy()

    scaler = StandardScaler()

    scaled_columns = scaler.fit_transform(df[continuous_cols])
    scaled_columns_test = scaler.transform(dft[continuous_cols])

    scaled_data = np.concatenate([df.drop(continuous_cols, axis=1).values, scaled_columns], axis=1)
    scaled_data_test = np.concatenate([dft.drop(continuous_cols, axis=1).values, scaled_columns_test], axis=1)

    if validation:
        dfv = df_val.copy()
        scaled_columns_val = scaler.transform(dfv[continuous_cols])
        scaled_data_val = np.concatenate([dfv.drop(continuous_cols, axis=1).values, scaled_columns_val], axis=1)
        return scaled_data, scaled_data_val, scaled_data_test

    return scaled_data, scaled_data_test


categorical_columns = ['sex', 'country_class', 'is_householder', 'tax_filer_class', 'race_class',
                       'occupation_class', 'is_married', 'college_degree', 'worker_type']
continuous_columns = ["age", "num persons worked for employer"]


def preprocess_data(train, test, validation_ratio=0.3, random_state=0, split=True):
    df, dft = train.copy(), test.copy()

    # Feature selection
    df_eng, test_eng = engineer_features(df), engineer_features(dft)

    # Split target and predictors
    X_train, X_test, y_train, y_test = df_eng.drop("target", axis=1), test_eng.drop("target", axis=1), df_eng[
        "target"].values, test_eng["target"].values

    # One hot encoding
    X_ohe, X_test_ohe = one_hot_encode_features(X_train, X_test, categorical_columns)

    if split:
        # Splitting training set into train and validation
        X_train, X_val, y_train, y_val = train_test_split(X_ohe, y_train, test_size=validation_ratio,
                                                          random_state=random_state,
                                                          stratify=y_train)

        # Scaling of continuous features
        X_train, X_val, X_test = scale_continuous(X_train, X_val, X_test_ohe, continuous_columns)

        return X_train, y_train, X_val, y_val, X_test, y_test

    else:
        # Scaling of continuous features
        X_train, X_test = scale_continuous(X_ohe, None, X_test_ohe, continuous_columns, validation=False)

        return X_train, y_train, X_test, y_test
