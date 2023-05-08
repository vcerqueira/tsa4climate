import pandas as pd
from plotnine import *
from mizani.formatters import date_format
from numerize import numerize

from src.acf import AutoCorrelation
from config import ASSETS, OUTPUTS

CHAPTER = 'Part 4'
assets = ASSETS[CHAPTER]
output_dir = OUTPUTS[CHAPTER]

file = f'{assets}/pjm/EKPC_hourly.csv'

# reading the data set
data = pd.read_csv(file, parse_dates=['Datetime'], index_col='Datetime')

series = data['EKPC_MW']
series.name = 'Demand'

series_df = series.reset_index()

# Average daily energy demand
series_df['Day'] = series.index.strftime('%Y-%m-%d')
daily_mean = series.resample('D').mean()
daily_mean.index = daily_mean.index.strftime('%Y-%m-%d')
daily_mean.to_dict()

# Average monthly energy demand
series_df['MonthlyAvg'] = series.index.strftime('%Y-%m')
monthly_mean = series.resample('M').mean()
monthly_mean.index = monthly_mean.index.strftime('%Y-%m')
monthly_mean.to_dict()

series_df['MonthlyAvg'] = series_df['MonthlyAvg'].map(monthly_mean)
series_df['DailyAvg'] = series_df['Day'].map(daily_mean)
series_df['Hour'] = series.index.hour
series_df['Month'] = series.index.month
series_df['Year'] = series.index.year
series_df['MonthPeriod'] = series.index.to_period('M')
series_df['Week'] = series.index.to_period('W')
series_df['Monthname'] = series.index.month_name()
series_df['Dayname'] = series.index.day_name()
series_df['Dayname'] = pd.Categorical(series_df['Dayname'],
                                      categories=['Monday', 'Tuesday', 'Wednesday',
                                                  'Thursday', 'Friday', 'Saturday',
                                                  'Sunday'])

series_df['Monthname'] = pd.Categorical(series_df['Monthname'],
                                        categories=['January', 'February', 'March',
                                                    'April', 'May', 'June', 'July',
                                                    'August', 'September', 'October',
                                                    'November', 'December'])

# TIME SERIES PLOT
demand_plot = ggplot(series_df) + \
              aes(x='Datetime', y='Demand') + \
              theme_classic(base_family='Palatino', base_size=12) + \
              theme(plot_margin=.125,
                    axis_text=element_text(size=11),
                    legend_title=element_blank(),
                    legend_position='top') + \
              geom_line(color='#541675') + \
              geom_line(mapping=aes(x='Datetime', y='DailyAvg'),
                        color='#f2c75b', size=1.3) + \
              xlab('') + \
              ylab('Energy Demand') + \
              ggtitle('') + \
              scale_y_continuous(labels=lambda lst: [numerize.numerize(x)
                                                     for x in lst])

# BOXPLOT ACROSS HOUR
daily_seasonality_plot = \
    ggplot(series_df) + \
    aes(x=0, y='Demand') + \
    theme_bw(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.125,
          axis_text_x=element_blank(),
          legend_title=element_blank(),
          strip_background_x=element_text(color='#f2c75b'),
          strip_text_x=element_text(size=11)) + \
    geom_boxplot() + \
    facet_grid('. ~Hour') + \
    labs(x='', y='Demand Distribution') + \
    scale_y_continuous(labels=lambda lst: [numerize.numerize(x)
                                           for x in lst])

# BOXPLOT ACROSS MONTH
monthly_distr_plot = \
    ggplot(series_df) + \
    aes(x=0, y='Demand') + \
    theme_bw(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.125,
          axis_text_x=element_blank(),
          legend_title=element_blank(),
          strip_background_x=element_text(color='#f2c75b'),
          strip_text_x=element_text(size=11)) + \
    geom_boxplot() + \
    facet_grid('. ~Monthname') + \
    labs(x='', y='Demand Distribution') + \
    scale_y_continuous(labels=lambda lst: [numerize.numerize(x)
                                           for x in lst])

# MONTHLY SEASONAL PLOT
monthly_seasonal_plot = \
    ggplot(series_df) + \
    aes(x='Monthname',
        y='MonthlyAvg',
        group='Year',
        color='Year'
        ) + \
    theme_classic(base_family='Palatino', base_size=14) + \
    theme(plot_margin=.15,
          # axis_text=element_text(size=12),
          axis_text_x=element_text(angle=30)) + \
    geom_line(size=1) + \
    scale_y_continuous(labels=lambda lst: [numerize.numerize(x) for x in lst]) + \
    labs(x='', y='Average daily demand')

# SEASONAL SUB-SERIES PLOT
stat_by_group = series_df.groupby('Monthname')['Demand'].mean()
stat_by_group = stat_by_group.reset_index()

# https://plotnine.readthedocs.io/en/stable/tutorials/miscellaneous-manipulating-date-breaks-and-date-labels.html
seasonal_subseries_plot = \
    ggplot(series_df) + \
    aes(x='Datetime',
        y='MonthlyAvg') + \
    theme_538(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.125,
          axis_text_x=element_text(size=8, angle=90),
          legend_title=element_blank(),
          strip_background_x=element_text(color='#f2c75b'),
          strip_text_x=element_text(size=11)) + \
    geom_line() + \
    facet_grid('. ~Monthname') + \
    geom_hline(data=stat_by_group,
               mapping=aes(yintercept='Demand'),
               colour='#f2c75b',
               size=2) + \
    labs(x='', y='Energy Demand') + \
    scale_x_datetime(labels=date_format('%Y'))

# HOURLY ACF PLOT
acorr = AutoCorrelation(n_lags=48, alpha=.05)
acorr.calc_acf(series)
acorr.acf_df['Type'] = 'Auto-correlation up to 48 lags'
acorr.acf_df['Laglabel'] = acorr.acf_df['Lag']
acf_series_plot = AutoCorrelation.acf_plot(acorr.acf_df) + facet_grid('~Type')

# DAILY ACF PLOT
acorr_day = AutoCorrelation(n_lags=365, alpha=.05)
acorr_day.calc_acf(series.resample('D').mean())
acorr_day.acf_df['Type'] = 'Auto-correlation up to 1 year of lags'
values_to_omit = acorr_day.acf_df['Lag'][[i % 14 != 0 for i in range(365 + 1)]].values
acorr_day.acf_df['Laglabel'] = ['' if x in values_to_omit else x for x in acorr_day.acf_df['Lag']]
acfd_series_plot = AutoCorrelation.acf_plot(acorr_day.acf_df) + facet_grid('~Type')

plot_size = {'height': 6, 'width': 13}

demand_plot.save(f'{output_dir}/1_demand_plot.pdf', **plot_size)
daily_seasonality_plot.save(f'{output_dir}/2_daily_seasonality_plot.pdf', **plot_size)
monthly_distr_plot.save(f'{output_dir}/3_monthly_dist_plot.pdf', **plot_size)
monthly_seasonal_plot.save(f'{output_dir}/4_monthly_seasonality_plot.pdf', **plot_size)
seasonal_subseries_plot.save(f'{output_dir}/5_monthly_ss_plot.pdf', **plot_size)
acf_series_plot.save(f'{output_dir}/6_acf_series_plot.pdf', **plot_size)
acfd_series_plot.save(f'{output_dir}/7_acfday_series_plot.pdf', **plot_size)
