import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from plotnine import *


class AutoCorrelation:
    PARAMS = {
        'bartlett_confint': True,
        'adjusted': False,
        'missing': 'none',
    }

    def __init__(self, n_lags: int, alpha: float):
        self.n_lags = n_lags
        self.alpha = alpha
        self.significance_thr = 0
        self.acf = None
        self.acf_df = None
        self.conf_int = None
        self.acf_analysis = {}

    def calc_acf(self, data: pd.Series):
        self.significance_thr = 2 / np.sqrt(len(data))

        acf_x = acf(
            data,
            nlags=self.n_lags,
            alpha=self.alpha,
            **self.PARAMS
        )

        self.acf, self.conf_int = acf_x[:2]

        self.acf_df = pd.DataFrame({
            'ACF': self.acf,
            'ACF_low': self.conf_int[:, 0],
            'ACF_high': self.conf_int[:, 1],
        })

        self.acf_df['Lag'] = ['t'] + [f't-{i}' for i in range(1, self.n_lags + 1)]
        self.acf_df['Lag'] = pd.Categorical(self.acf_df['Lag'], categories=self.acf_df['Lag'])

    @staticmethod
    def acf_plot(acf_data: pd.DataFrame):
        plot = ggplot(acf_data, aes(x='Lag', y='ACF'))
        plot += geom_hline(yintercept=0, linetype='solid', color='black', size=1)
        plot = \
            plot + geom_segment(
                aes(x='Lag',
                    xend='Lag',
                    y=0,
                    yend='ACF'),
                size=1.5,
                color='#b193c4'
            ) + \
            geom_point(
                size=4,
                color='#460076',
            ) + \
            theme_bw(base_family='Palatino', base_size=12) + \
            theme(plot_margin=.125,
                  axis_text_y=element_text(size=14),
                  axis_text_x=element_text(size=10, angle=90),
                  legend_title=element_blank(),
                  legend_position='none') + \
            scale_x_discrete(breaks=acf_data['Lag'], labels=acf_data['Laglabel'])

        plot = plot + \
               xlab('') + \
               ylab('Auto-correlation') + \
               ylim(-1, 1) + \
               ggtitle('')

        return plot
