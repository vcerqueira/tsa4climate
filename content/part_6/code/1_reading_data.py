import pandas as pd
from plotnine import *

from config import ASSETS, OUTPUTS

PART = 'Part 6'
assets = ASSETS[PART]
output_dir = OUTPUTS[PART]

DATE_TIME_COLS = ['month', 'day', 'calendar_year', 'hour', 'water_year']

file = f'{assets}/dewpoint_final.csv'

# reading the data set
data = pd.read_csv(file)

# parsing the datetime column
data['datetime'] = \
    pd.to_datetime([f'{year}/{month}/{day} {hour}:00'
                    for year, month, day, hour in zip(data['calendar_year'],
                                                      data['month'],
                                                      data['day'],
                                                      data['hour'])])

data = data.drop(DATE_TIME_COLS, axis=1).set_index('datetime')
data.columns = data.columns.str.replace('_dpt_C', '')

df = data.reset_index().melt('datetime')

plot = \
    ggplot(df) + \
    aes(x='datetime',
        y='value',
        group='variable',
        color='variable') + \
    theme_classic(base_family='Palatino', base_size=14) + \
    theme(plot_margin=0.2,
          axis_text_y=element_text(size=12),
          axis_text_x=element_text(angle=0, size=12),
          legend_title=element_blank(),
          legend_position='top') + \
    geom_line(size=1, alpha=0.7) + \
    labs(x='',
         y='Dew point',
         title='')

print(plot)

corr_mat = data.corr().reset_index().melt('index')

plot_mat = ggplot(corr_mat, aes('index', 'variable', fill='value')) + \
           geom_tile(aes(width=.95, height=.95)) + \
           theme_light() + \
           theme(plot_margin=.2,
                 axis_ticks=element_blank(),
                 axis_text_x=element_text(angle=90),
                 panel_background=element_rect(fill='white')) + \
           labs(x='', y='')

print(plot_mat)
