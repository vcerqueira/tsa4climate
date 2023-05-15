import pandas as pd
from plotnine import *
from numerize import numerize

from src.log import LogTransformation

from config import ASSETS, OUTPUTS

PART = 'Part 5'
assets = ASSETS[PART]
output_dir = OUTPUTS[PART]

TARGET_EVENTS = ['Hail', 'Thunderstorm Wind']
METEOROLOGICAL_DATA = ['WSPD', 'WVHT', 'PRES', 'WTMP', 'APD']
N_LAGS = 4
HORIZON = 12

storms = pd.read_csv(f'{assets}/storms_data.csv', index_col='storm_start')
storms.index = pd.to_datetime(storms.index)

buoys = pd.read_csv(f'{assets}/buoys.csv', index_col='datetime')
buoys.index = pd.to_datetime(buoys.index).tz_localize('UTC')
buoys['STATION'] = buoys['STATION'].astype(str)

buoys_h = buoys.groupby('STATION').resample('H').mean()
buoys_h = buoys_h[METEOROLOGICAL_DATA]

buoys_df = buoys_h.reset_index('STATION')

station_df = buoys_df.loc[buoys_df['STATION'] == '41009']
station_df = station_df.drop('STATION', axis=1)

df = station_df.tail(10000).reset_index().melt('datetime')
df['log_value'] = LogTransformation.transform(df['value'])

plot = \
    ggplot(df) + \
    aes(x='datetime',
        y='log_value',
        group='variable',
        color='variable') + \
    theme_classic(base_family='Palatino', base_size=14) + \
    theme(plot_margin=0.2,
          axis_text_y=element_text(size=12),
          axis_text_x=element_text(angle=0, size=12),
          legend_title=element_blank(),
          legend_position='top') + \
    geom_line(size=1) + \
    labs(x='',
         y='Value (log-scale)',
         title='')

costs_by_year = storms.groupby(storms.index.year).sum()
costs_by_year = costs_by_year.reset_index()[-24:]
costs_by_year['storm_start'] = costs_by_year['storm_start'].astype(str)
costs_by_year.columns = ['storm_start',
                         'Damage to Property',
                         'Damage to Crops']
df = costs_by_year.melt('storm_start')

costs_plot = ggplot(df) + \
             aes(x='storm_start',
                 y='value',
                 fill='variable',
                 group='variable') + \
             theme_classic(base_family='Palatino', base_size=12) + \
             theme(plot_margin=.15,
                   axis_text=element_text(size=12),
                   axis_text_x=element_text(angle=90),
                   legend_title=element_blank(),
                   legend_position='top') + \
             geom_bar(stat='identity',
                      position='dodge',
                      width=1) + \
             xlab('') + \
             ylab('Estimated costs') + \
             ggtitle('') + \
             scale_fill_manual(values=['#111e6c', '#4f97a3']) + \
             scale_y_continuous(labels=lambda lst: [numerize.numerize(x)
                                                    for x in lst])

print(costs_plot)

top_20_events = storms['EVENT_TYPE'].value_counts()[:20]
top_20_events = top_20_events.reset_index()
top_20_events.columns = ['Event', 'Occurrences']
top_20_events['Event'] = pd.Categorical(top_20_events['Event'],
                                        categories=top_20_events['Event'])

plot_events = ggplot(top_20_events) + \
              aes(x='Event',
                  y='Occurrences') + \
              theme_classic(base_family='Palatino', base_size=12) + \
              theme(plot_margin=.175,
                    axis_text=element_text(size=13),
                    axis_text_x=element_text(angle=60, size=10),
                    legend_title=element_blank(),
                    legend_position='top') + \
              geom_bar(stat='identity',
                       position='dodge',
                       width=.8,
                       fill='#4f97a3') + \
              xlab('') + \
              ylab('No. of events') + \
              ggtitle('')

print(plot_events)

storms.groupby(storms.index.year).count()

events = storms.loc[storms['EVENT_TYPE'].isin(TARGET_EVENTS), :]

no_of_events = events.index.year.value_counts().sort_index()[:-1]
no_of_events = no_of_events.reset_index()
no_of_events.columns = ['Year', 'No. of events']
# no_of_events['Year'] = pd.Categorical(no_of_events['Year'],
#                                       categories=no_of_events['Year'])

plot_no_events = ggplot(no_of_events) + \
                 aes(x='Year',
                     y='No. of events') + \
                 theme_classic(base_family='Palatino', base_size=12) + \
                 theme(plot_margin=.175,
                       axis_text=element_text(size=13),
                       # axis_text_x=element_text(angle=60, size=10),
                       legend_title=element_blank(),
                       legend_position='top') + \
                 geom_bar(stat='identity',
                          position='dodge',
                          width=.8,
                          fill='#4f97a3') + \
                 xlab('') + \
                 ylab('No. of events') + \
                 ggtitle('')

print(plot_no_events)
