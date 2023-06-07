import numpy as np
import pandas as pd
from plotnine import *

from config import ASSETS, OUTPUTS

PART = 'Part 7'
assets = ASSETS[PART]
output_dir = OUTPUTS[PART]

EXCLUDE_CATEGORIES = ['Alcohol', 'All Foods', 'Other']
TARGET_VAR = 'Dollars'

file = f'{assets}/NationalTotalAndSubcategory.csv'

# reading the data set
data = pd.read_csv(file)

pd.set_option('display.max_columns', None)

data = data.loc[data['variable'] == TARGET_VAR, :]
data = data.loc[~data['Category'].isin(EXCLUDE_CATEGORIES), :]
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data.set_index('Date', inplace=True)

data['ID'] = [f'{x}_{y}' for x, y in zip(data['Category'], data['Subcategory'])]

unique_prods = data.ID.unique().tolist()
prod_series = {}
for prod in unique_prods:
    prod_series[prod] = data.loc[data['ID'] == prod]['value'] / 1_000_000

prod_df = pd.DataFrame(prod_series)

prod_df = prod_df.sort_index()

prod_df.to_csv(f'{assets}/food_sales.csv')

df = prod_df.reset_index().melt('Date')

df['Category'] = df['variable'].apply(lambda x: x.split('_')[0])

df['log_value'] = np.log(df['value'] + 1)

plot = \
    ggplot(df) + \
    aes(x='Date',
        y='log_value',
        group='variable',
        color='Category') + \
    theme_classic(base_family='Palatino', base_size=14) + \
    theme(plot_margin=0.2,
          axis_text_y=element_text(size=12),
          axis_text_x=element_text(angle=0, size=11),
          legend_title=element_blank(),
          legend_position='top') + \
    geom_line(size=1) + \
    labs(x='',
         y='Log sales (millions of $)',
         title='')

corr_mat = prod_df.corr().reset_index().melt('index')

plot_mat = ggplot(corr_mat, aes('index', 'variable', fill='value')) + \
           geom_tile(aes(width=.95, height=.95)) + \
           theme_light() + \
           theme(plot_margin=.2,
                 axis_ticks=element_blank(),
                 axis_text_x=element_text(angle=90),
                 panel_background=element_rect(fill='white')) + \
           labs(x='', y='')
