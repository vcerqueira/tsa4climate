import re

import pandas as pd
from plotnine import *

from config import ASSETS, OUTPUTS
from src.log import LogTransformation

PART = 'Part 2'
assets = ASSETS[PART]
output_dir = OUTPUTS[PART]

DATE_TIME_COLS = ['month', 'day', 'calendar_year', 'hour']
STATION = 'smf1'

COLUMNS_PER_FILE = \
    {'incoming_solar_final.csv': DATE_TIME_COLS + [f'{STATION}_sin_w/m2'],
     'wind_dir_raw.csv': DATE_TIME_COLS + [f'{STATION}_wd_deg'],
     'snow_depth_final.csv': DATE_TIME_COLS + [f'{STATION}_sd_mm'],
     'wind_speed_final.csv': DATE_TIME_COLS + [f'{STATION}_ws_m/s'],
     'dewpoint_final.csv': DATE_TIME_COLS + [f'{STATION}_dpt_C'],
     'precipitation_final.csv': DATE_TIME_COLS + [f'{STATION}_ppt_mm'],
     'vapor_pressure.csv': DATE_TIME_COLS + [f'{STATION}_vp_Pa'],
     'relative_humidity_final.csv': DATE_TIME_COLS + [f'{STATION}_rh'],
     'air_temp_final.csv': DATE_TIME_COLS + [f'{STATION}_ta_C'],
     }

data_series = {}
for file in COLUMNS_PER_FILE:

    file_data = pd.read_csv(f'{assets}/{file}')

    var_df = file_data[COLUMNS_PER_FILE[file]]

    var_df['datetime'] = \
        pd.to_datetime([f'{year}/{month}/{day} {hour}:00'
                        for year, month, day, hour in zip(var_df['calendar_year'],
                                                          var_df['month'],
                                                          var_df['day'],
                                                          var_df['hour'])])

    var_df = var_df.drop(DATE_TIME_COLS, axis=1)
    var_df = var_df.set_index('datetime')
    series = var_df.iloc[:, 0].sort_index()

    data_series[file] = series

mv_series = pd.concat(data_series, axis=1)
mv_series.columns = [re.sub('_final.csv|_raw.csv|.csv', '', x) for x in mv_series.columns]
mv_series.columns = [re.sub('_', ' ', x) for x in mv_series.columns]
mv_series.columns = [x.title() for x in mv_series.columns]

mv_series = mv_series.astype(float)

## plotting

mv_series_d = mv_series.resample('D').mean()
mv_series_df = mv_series_d.reset_index().melt('datetime')
mv_series_df['log_value'] = LogTransformation.transform(mv_series_df['value'])

plot = \
    ggplot(mv_series_df) + \
    aes(x='datetime',
        y='log_value',
        group='variable',
        color='variable') + \
    theme_classic(base_family='Palatino', base_size=12) + \
    theme(plot_margin=0.2,
          axis_text_y=element_text(size=10),
          axis_text_x=element_text(angle=0, size=9),
          legend_title=element_blank(),
          legend_position='top') + \
    geom_line() + \
    labs(x='',
         y='Value (log-scale)',
         title='')

plot_unv = \
    ggplot(mv_series.resample('D').sum().reset_index()) + \
    aes(x='datetime', y='Incoming Solar', group=1) + \
    theme_classic(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.125,
          axis_text=element_text(size=12),
          legend_title=element_blank(),
          legend_position=None)

plot_unv += geom_line(color='#f58216', size=1)

plot_unv = \
    plot_unv + \
    xlab('') + \
    ylab('Solar Irradiance (w/m2)') + \
    ggtitle('')

data_corr = mv_series.corr()
data_corr = data_corr.reset_index().melt('index')
data_corr.columns = ['var1', 'var2', 'value']
data_corr['lab'] = data_corr['value'].round(2).astype(str)

plot_corr = \
    ggplot(data_corr, aes(x='var1', y='var2', fill='value')) + \
    geom_tile() + \
    geom_label(aes(label='lab'),
               fill='white',
               size=8) + \
    scale_fill_gradient2(low='darkorange', mid='white', high='steelblue') + \
    theme_classic() + \
    labs(x='', y='', title='') + \
    theme(plot_margin=0.2,
          axis_text_x=element_text(rotation=45, hjust=1),
          axis_ticks=element_blank())

plot.save(f'{output_dir}/mv_line_plot.pdf', height=5, width=9)
plot_unv.save(f'{output_dir}/uv_line_plot.pdf', height=4, width=9)
plot_corr.save(f'{output_dir}/corr_plot.pdf', height=6, width=6)
