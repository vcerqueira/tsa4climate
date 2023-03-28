import datetime

import numpy as np
import pandas as pd
from plotnine import *

from config import CHAPTER_ASSETS, CHAPTER_OUTPUTS

CHAPTER = 'Chapter 3'
assets = CHAPTER_ASSETS[CHAPTER]
output_dir = CHAPTER_OUTPUTS[CHAPTER]

data = pd.read_csv(f'{assets}/wind_df.csv', parse_dates=['datetime'], index_col='datetime')
data.drop('rec_fcast', axis=1, inplace=True)
data = data.loc[[2013 < x < 2018 for x in data.index.year], :]
data.columns = ['Wind Power', 'Inst. Capacity']

data_melted = data.reset_index().melt('datetime')

plot = \
    ggplot(data_melted) + \
    aes(x='datetime',
        y='value',
        group='variable',
        color='variable') + \
    theme_classic(base_family='Palatino', base_size=12) + \
    theme(plot_margin=0.2,
          axis_text_y=element_text(size=11),
          axis_text_x=element_text(angle=0, size=10),
          legend_title=element_blank(),
          legend_position='top') + \
    geom_line(size=1) + \
    labs(x='', y='value', title='') + \
    scale_color_manual(values=['#d92121', '#444444'])

data['Norm. Wind Power'] = data['Wind Power'] / data['Inst. Capacity']

series = data['Norm. Wind Power'].resample('H').mean()
series.iloc[np.where(series > 1)] = np.nan
series = series.ffill()

series_df_d = series.resample('D').mean().reset_index()
series_df_h_sample = series[:(24 * 93)].reset_index()
series_df_h_dt = series[:(24 * 93)].diff().reset_index()
# VER ISTO, DEVIA USAR % NAO?
change_thr = 0.1

# daily series. higher granularity for visualization
plot_uv_d = ggplot(series_df_d) + \
            aes(x='datetime', y='Norm. Wind Power', group=1) + \
            theme_classic(base_family='Palatino', base_size=12) + \
            theme(plot_margin=.15,
                  axis_text=element_text(size=12),
                  legend_title=element_blank(),
                  legend_position='top') + \
            geom_line(size=0.9, color='#555555') + \
            xlab('') + \
            ylab('Normalized Wind Power') + \
            ggtitle('')

# first four months of hourly data
plot_uv_h4m = ggplot(series_df_h_sample) + \
              aes(x='datetime', y='Norm. Wind Power', group=1) + \
              theme_classic(base_family='Palatino', base_size=12) + \
              theme(plot_margin=.15,
                    axis_text=element_text(size=12),
                    legend_title=element_blank(),
                    legend_position='top') + \
              geom_line(size=0.9, color='#555555') + \
              xlab('') + \
              ylab('Normalized Wind Power') + \
              ggtitle('')

# same 4 months but series of changes
plot_uv_h4m_dt = ggplot(series_df_h_dt) + \
                 aes(x='datetime', y='Norm. Wind Power', group=1) + \
                 theme_classic(base_family='Palatino', base_size=12) + \
                 theme(plot_margin=.15,
                       axis_text=element_text(size=12),
                       legend_title=element_blank(),
                       legend_position='top') + \
                 geom_line(size=0.9, color='#555555') + \
                 geom_hline(yintercept=change_thr, linetype='dashed', color='red', size=1.1) + \
                 geom_hline(yintercept=-change_thr, linetype='dashed', color='red', size=1.1) + \
                 xlab('') + \
                 ylab('Hourly Changes in Wind Power') + \
                 ggtitle('')

series_above_thr = series.diff() > change_thr
series_below_thr = series.diff() < -change_thr

series_above_thr_month = series_above_thr.groupby(series.index.month).mean() * 100
series_above_thr_month.index = \
    [datetime.datetime.strptime(str(x), "%m").strftime("%B")
     for x in series_above_thr_month.index]

series_above_thr_month = series_above_thr_month.reset_index()
series_above_thr_month['Ramp Event'] = 'Upward Ramp'

series_below_thr_month = series_below_thr.groupby(series.index.month).mean() * 100
series_below_thr_month.index = \
    [datetime.datetime.strptime(str(x), "%m").strftime("%B")
     for x in series_below_thr_month.index]

series_below_thr_month = series_below_thr_month.reset_index()
series_below_thr_month['Ramp Event'] = 'Downward Ramp'

ramp_perc_p_month = pd.concat([series_above_thr_month, series_below_thr_month], ignore_index=True)

month_list = pd.Categorical(ramp_perc_p_month['index'], categories=ramp_perc_p_month['index'].unique())
ramp_perc_p_month = ramp_perc_p_month.assign(month_list=month_list)

# ramp event probability per month
plot_ramps_by_month = ggplot(ramp_perc_p_month) + \
                      aes(x='month_list',
                          y='Norm. Wind Power',
                          fill='Ramp Event',
                          group='Ramp Event') + \
                      theme_classic(base_family='Palatino',
                                    base_size=12) + \
                      theme(plot_margin=.15,
                            axis_text=element_text(size=11),
                            legend_title=element_blank(),
                            legend_position='top') + \
                      geom_bar(stat='identity', position='dodge') + \
                      xlab('') + \
                      ylab('Ramp Event Probability') + \
                      ggtitle('')

plot.save(f'{output_dir}/mv_line.pdf', height=5, width=12)
plot_uv_d.save(f'{output_dir}/daily_line.pdf', height=5, width=12)
plot_uv_h4m.save(f'{output_dir}/hourly_line_sample.pdf', height=5, width=12)
plot_uv_h4m_dt.save(f'{output_dir}/hourly_line_changes.pdf', height=5, width=12)
plot_ramps_by_month.save(f'{output_dir}/ramps_by_month.pdf', height=5, width=12)
