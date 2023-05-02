from datetime import datetime

import numpy as np
import pandas as pd


class FourierTerms:

    def __init__(self, period: float, n_terms: int, prefix=''):
        self.period = period
        self.n_terms = n_terms
        self.prefix = prefix

    def transform(self, index: pd.DatetimeIndex, use_as_index: bool = True):
        t = np.array(
            (index - datetime(1970, 1, 1)).total_seconds().astype(float)
        ) / (3600 * 24.)

        fourier_x = np.column_stack([
            fun((2.0 * (i + 1) * np.pi * t / self.period))
            for i in range(self.n_terms)
            for fun in (np.sin, np.cos)
        ])

        col_names = [
            f'{self.prefix}{fun.__name__[0].upper()}{i}'
            for i in range(self.n_terms)
            for fun in (np.sin, np.cos)
        ]

        fourier_df = pd.DataFrame(fourier_x, columns=col_names)

        if use_as_index:
            fourier_df.index = index

        return fourier_df
