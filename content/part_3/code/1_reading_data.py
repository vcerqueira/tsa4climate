import pandas as pd
from plotnine import *

from config import ASSETS, OUTPUTS

CHAPTER = 'Part 3'
assets = ASSETS[CHAPTER]
output_dir = OUTPUTS[CHAPTER]

file = f'{assets}/sample_IrishSmartBuoy.csv'

# reading the data set
# skipping the second with skiprows
# parsing time column to datetime and setting it as index
data = pd.read_csv(file, skiprows=[1], parse_dates=['time'], index_col='time')

# defining the series and converting cm to meters
series = data['SignificantWaveHeight'] / 100
# resampling to hourly and taking the mean
series = series.resample('H').mean()

series_df = series.reset_index()

plot_line = ggplot(series_df) + \
            aes(x='time', y='SignificantWaveHeight') + \
            theme_538(base_family='Palatino', base_size=12) + \
            theme(plot_margin=.175,
                  axis_text_y=element_text(size=12),
                  axis_text_x=element_text(size=9)) + \
            geom_line(color='#192841', size=.85) + \
            xlab('') + \
            ylab('Sign. Wave Height (meters)')

plot_hist = ggplot(series_df) + \
            aes(x='SignificantWaveHeight') + \
            theme_538(base_family='Palatino', base_size=12) + \
            theme(plot_margin=.15,
                  axis_text=element_text(size=12),
                  legend_title=element_blank(),
                  legend_position='top') + \
            geom_histogram(alpha=.9,
                           bins=50,
                           color='#192841',
                           fill='#192841') + \
            xlab('Significant Wave Height') + \
            ylab('Frequency') + \
            geom_vline(xintercept=5,
                       linetype='dashed',
                       color='#fee12b',
                       size=1.3)

plot_line.save(f'{output_dir}/wave_plot.pdf', height=6, width=12)
plot_hist.save(f'{output_dir}/waves_dist.pdf', height=5, width=8)
