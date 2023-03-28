import numpy as np
import pandas as pd


def correlation_filter(data: pd.DataFrame, corr_threshold: float = .9):
    # Absolute correlation matrix
    corr_matrix = data.corr().abs()

    # Create a True/False mask and apply it
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    tri_df = corr_matrix.mask(mask)

    # List column names of highly correlated features (r > 0.95)
    corr_features = \
        [c for c in tri_df.columns
         if any(tri_df[c] > corr_threshold)]

    # Drop the features in the to_drop list
    data_subset = data.drop(corr_features, axis=1)

    return data_subset

