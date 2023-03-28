import re

import numpy as np
import pandas as pd
from sktime.transformations.series.date import DateTimeFeatures


def feature_engineering(X: pd.DataFrame, y=None) -> pd.DataFrame:
    """
    param X: lagged observations (explanatory variables)

    :return: new features
    """

    summary_stats = {'mean': np.mean, 'sdev': np.std}

    features = {}
    for f in summary_stats:
        features[f] = X.apply(lambda x: summary_stats[f](x), axis=1)

    features_df = pd.concat(features, axis=1)
    X_feats = pd.concat([X, features_df], axis=1)

    return X_feats

#
# def volatility_summary(X: pd.DataFrame):
#     """
#     Feature extraction to summarise volatility
#
#     :param X: lag variables
#     :return: X plus extra features
#     """
#
#     lag_changes = X.apply(lambda x: x.diff()[1:], axis=1)
#     lag_changes.columns = [f'D_{x}' for x in lag_changes.columns]
#
#     summary_changes = {
#         'dt_std': lag_changes.std(axis=1),
#         'dt_avg': lag_changes.mean(axis=1),
#     }
#
#     summary_changes_df = pd.DataFrame(summary_changes)
#
#     feature_set = pd.concat([X, lag_changes, summary_changes_df], axis=1)
#
#     return feature_set
