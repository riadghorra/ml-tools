import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_continuous_column_against_target(data, column_name, target, limit=None):
    x = data[column_name]
    y = data[target]
    if limit:
        r = data[data[column_name] < limit]
        x = r[column_name]
        y = r[target]

    plt.scatter(x, y)
    plt.show()


def identify_missing(data, missing_threshold):
    """Find the features with a fraction of missing values above `missing_threshold`"""

    missing_threshold = missing_threshold

    # Calculate the fraction of missing in each column
    missing_series = data.isnull().sum() / data.shape[0]
    missing_stats = pd.DataFrame(missing_series).rename(columns={'index': 'feature', 0: 'missing_fraction'})

    # Sort with highest number of missing values on top
    missing_stats = missing_stats.sort_values('missing_fraction', ascending=False)

    # Find the columns with a missing percentage above the threshold
    record_missing = pd.DataFrame(missing_series[missing_series > missing_threshold]).reset_index().rename(columns=
    {
        'index': 'feature',
        0: 'missing_fraction'})

    to_drop = list(record_missing['feature'])

    record_missing = record_missing

    print('%d features with greater than %0.2f missing values.\n' % (len(to_drop), missing_threshold))
    return missing_stats


def identify_collinear(data, correlation_threshold, one_hot=False):
    """
    Finds collinear features based on the correlation coefficient between features.
    For each pair of features with a correlation coefficient greater than `correlation_threshold`,
    only one of the pair is identified for removal.
    Using code adapted from: https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/

    Parameters
    --------
    data : pandas dataframe
    correlation_threshold : float between 0 and 1
        Value of the Pearson correlation coefficient for identifying correlation features
    one_hot : boolean, default = False
        Whether to one-hot encode the features before calculating the correlation coefficients
    """
    data_all = data.copy()

    # Calculate the correlations between every column
    if one_hot:

        # One hot encoding
        features = pd.get_dummies(data)
        one_hot_features = [column for column in features.columns if column not in data.columns]

        # Add one hot encoded data to original data
        data_all = pd.concat([features[one_hot_features], data], axis=1)

        corr_matrix = pd.get_dummies(features).corr()

    else:
        corr_matrix = data.corr()

    corr_matrix = corr_matrix

    # Extract the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Select the features with correlations above the threshold
    # Need to use the absolute value
    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

    # Dataframe to hold correlated pairs
    record_collinear = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])

    # Iterate through the columns to drop to record pairs of correlated features
    for column in to_drop:
        # Find the correlated features
        corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

        # Find the correlated values
        corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
        drop_features = [column for _ in range(len(corr_features))]

        # Record the information (need a temp df for now)
        temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                          'corr_feature': corr_features,
                                          'corr_value': corr_values})

        # Add to dataframe
        record_collinear = record_collinear.append(temp_df, ignore_index=True)

    print('%d features with a correlation magnitude greater than %0.2f.\n' % (len(to_drop), correlation_threshold))
    return record_collinear, data_all.drop(to_drop, axis=1)


def identify_single_unique(data):
    """Finds features with only a single unique value. NaNs do not count as a unique value. """

    # Calculate the unique counts in each column
    unique_counts = data.nunique()

    # Find the columns with only one unique count
    record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(
        columns={'index': 'feature',
                 0: 'nunique'})

    to_drop = list(record_single_unique['feature'])

    print('%d features with a single unique value.\n' % len(to_drop))
    return record_single_unique


def plot_correlation_matrix_and_target_correlation(data, target, one_hot=False):
    if one_hot:

        # One hot encoding
        features = pd.get_dummies(data)

        # Add one hot encoded data to original data
        corr_matrix = pd.get_dummies(features).corr()

    else:
        corr_matrix = data.corr()

    plt.figure(figsize=(22, 20))
    sns.heatmap(corr_matrix.drop(target, axis=1), annot=True, cmap=plt.cm.Reds)
    plt.show()

    # Correlation with output variable
    relevant_features = abs(corr_matrix[target]).to_frame().sort_values(target, ascending=False).reset_index().rename(
        columns={target: "correlation with {}".format(target), "index": "feature"}).drop(0)
    return relevant_features
