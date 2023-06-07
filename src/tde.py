import re

import pandas as pd
import numpy as np


def time_delay_embedding(series: pd.Series,
                         n_lags: int,
                         horizon: int,
                         return_Xy: bool = False):
    """
    Time delay embedding
    Time series for supervised learning

    :param series: time series as pd.Series
    :param n_lags: number of past values to used as explanatory variables
    :param horizon: how many values to forecast
    :param return_Xy: whether to return the lags split from future observations

    :return: pd.DataFrame with reconstructed time series
    """
    assert isinstance(series, pd.Series)

    if series.name is None:
        name = 'Series'
    else:
        name = series.name

    n_lags_iter = list(range(n_lags, -horizon, -1))

    df_list = [series.shift(i) for i in n_lags_iter]
    df = pd.concat(df_list, axis=1).dropna()
    df.columns = [f'{name}(t-{j - 1})'
                  if j > 0 else f'{name}(t+{np.abs(j) + 1})'
                  for j in n_lags_iter]

    df.columns = [re.sub('t-0', 't', x) for x in df.columns]

    if not return_Xy:
        return df

    is_future = df.columns.str.contains('\+')

    X = df.iloc[:, ~is_future]
    Y = df.iloc[:, is_future]
    if Y.shape[1] == 1:
        Y = Y.iloc[:, 0]

    return X, Y


def mts_to_tabular(data: pd.DataFrame,
                   n_lags: int,
                   horizon: int,
                   return_Xy: bool = False,
                   drop_na: bool = True):
    """
    Time delay embedding with multivariate time series
    Time series for supervised learning

    :param data: multivariate time series as pd.DataFrame
    :param n_lags: number of past values to used as explanatory variables
    :param horizon: how many values to forecast
    :param return_Xy: whether to return the lags split from future observations

    :return: pd.DataFrame with reconstructed time series
    """

    data_list = [time_delay_embedding(data[col], n_lags, horizon)
                 for col in data]

    df = pd.concat(data_list, axis=1)

    if drop_na:
        df = df.dropna()

    if not return_Xy:
        return df

    is_future = df.columns.str.contains('\+')

    X = df.iloc[:, ~is_future]
    Y = df.iloc[:, is_future]

    if Y.shape[1] == 1:
        Y = Y.iloc[:, 0]

    return X, Y


def from_matrix_to_3d(df: pd.DataFrame) -> np.ndarray:
    """
    Transforming a time series from matrix into 3-d structure for deep learning

    :param df: (pd.DataFrame) Time series in the matrix format after embedding

    :return: Reshaped time series into 3-d structure
    """

    cols = df.columns

    # getting unique variables in the time series
    # this list has a single element for univariate time series
    var_names = np.unique([re.sub(r'\([^)]*\)', '', c) for c in cols]).tolist()

    # getting observation for each variable
    arr_by_var = [df.loc[:, cols.str.contains(v)].values for v in var_names]
    # reshaping the data of each variable into a 3-d format
    arr_by_var = [x.reshape(x.shape[0], x.shape[1], 1) for x in arr_by_var]

    # concatenating the arrays of each variable into a single array
    ts_arr = np.concatenate(arr_by_var, axis=2)

    return ts_arr


def from_3d_to_matrix(arr: np.ndarray, col_names: pd.Index):
    if arr.shape[2] > 1:
        arr_split = np.dsplit(arr, arr.shape[2])
    else:
        arr_split = [arr]

    arr_reshaped = [x.reshape(x.shape[0], x.shape[1]) for x in arr_split]

    df = pd.concat([pd.DataFrame(x) for x in arr_reshaped], axis=1)

    df.columns = col_names

    return df


def transform_mv_series(dataset: pd.DataFrame, n_lags: int, horizon: int):
    # preparing the time series for supervised learning
    # transforming each variable into a matrix format
    mat_by_variable = []
    for col in dataset:
        col_df = time_delay_embedding(dataset[col], n_lags=n_lags, horizon=horizon)
        mat_by_variable.append(col_df)

    # concatenating all variables
    mat_df = pd.concat(mat_by_variable, axis=1).dropna()

    # defining target (Y) and explanatory variables (X)
    predictor_variables = mat_df.columns.str.contains('\(t\-|\(t\)')
    target_variables = mat_df.columns.str.contains('\(t\+')
    X = mat_df.iloc[:, predictor_variables]
    Y = mat_df.iloc[:, target_variables]
    target_colnames = Y.columns

    X_3d = from_matrix_to_3d(X)
    Y_3d = from_matrix_to_3d(Y)

    return X_3d, Y_3d, target_colnames
