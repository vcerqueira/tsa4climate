import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sktime.transformations.series.date import DateTimeFeatures
from plotnine import *

from src.tde import time_delay_embedding
from src.fourier import FourierTerms
from src.error import mase
from config import ASSETS, OUTPUTS

CHAPTER = 'Part 4'
assets = ASSETS[CHAPTER]
output_dir = OUTPUTS[CHAPTER]

file = f'{assets}/pjm/EKPC_hourly.csv'

# Reading the data set
data = pd.read_csv(file, parse_dates=['Datetime'], index_col='Datetime')

series = data['EKPC_MW']
series.name = 'Demand'
series = series.resample('H').mean()

# Modeling

# Train / test split
train, test = train_test_split(series, test_size=0.2, shuffle=False)

# using past 12 observations as explanatory variables
N_LAGS = 12
# using the next 12 hours as the forecasting horizon
HORIZON = 12

# transforming time series into a tabular format for supervised learning
X_train, Y_train = time_delay_embedding(train, n_lags=N_LAGS, horizon=HORIZON, return_Xy=True)
X_test, Y_test = time_delay_embedding(test, n_lags=N_LAGS, horizon=HORIZON, return_Xy=True)

hourly_feats = DateTimeFeatures(ts_freq='H',
                                keep_original_columns=False,
                                feature_scope='efficient')

fourier_daily = FourierTerms(n_terms=2, period=24, prefix='D_')
fourier_monthly = FourierTerms(n_terms=2, period=24 * 30.5, prefix='M_')
fourier_yearly = FourierTerms(n_terms=2, period=24 * 365, prefix='Y_')

dtime_train = hourly_feats.fit_transform(X_train)
dfourier_train = fourier_daily.transform(X_train.index)
mfourier_train = fourier_monthly.transform(X_train.index)
yfourier_train = fourier_yearly.transform(X_train.index)

feats_train = pd.concat([X_train, dtime_train, dfourier_train,
                         mfourier_train, yfourier_train],
                        axis=1)

model = RandomForestRegressor()
model.fit(feats_train, Y_train)

dtime_test = hourly_feats.transform(X_test)
dfourier_test = fourier_daily.transform(X_test.index)
mfourier_test = fourier_monthly.transform(X_test.index)
yfourier_test = fourier_yearly.transform(X_test.index)

feats_test = pd.concat([X_test, dtime_test, dfourier_test,
                        mfourier_test, yfourier_test],
                       axis=1)

preds = model.predict(feats_test)
preds = pd.DataFrame(preds, columns=Y_test.columns)

mase_scores = pd.Series({c: mase(training_series=Y_train[c].values,
                                 testing_series=Y_test[c].values,
                                 prediction_series=preds[c].values)
                         for c in preds.columns})

mase_scores.plot()

results = mase_scores.reset_index()
results.columns = ['Horizon', 'Score']
results['Horizon'] = [s[s.find("(") + 1:s.find(")")] for s in results['Horizon']]
results['Horizon'] = pd.Categorical(results['Horizon'], categories=results['Horizon'])
results['alpha'] = results['Score'] + .5

plt_results = ggplot(results) + \
              aes(x='Horizon', y='Score', fill='Score') + \
              theme_classic(base_family='Palatino', base_size=14) + \
              theme(plot_margin=.2,
                    axis_text_y=element_text(size=14),
                    axis_text_x=element_text(size=12)) + \
              geom_bar(stat='identity',
                       position='dodge') + \
              xlab('') + \
              ylab('MASE error') + \
              scale_fill_gradient2(low='#614b00',
                                   mid='#ffd037',
                                   high='#ffe9a1') + \
              guides(fill=False)

importance_scores = pd.Series(dict(zip(feats_train.columns, model.feature_importances_)))

imp_df = importance_scores.reset_index()
imp_df.columns = ['Feature', 'Importance']
imp_df = imp_df.sort_values('Importance', ascending=False)
imp_df['Feature'] = pd.Categorical(imp_df['Feature'], categories=imp_df['Feature'])

plot = ggplot(imp_df[:20], aes(x='Feature', y='Importance')) + \
       geom_bar(fill='#c49102', stat='identity', position='dodge') + \
       theme_classic(
           base_family='Palatino',
           base_size=12) + \
       theme(
           plot_margin=.2,
           axis_text=element_text(size=12),
           axis_text_x=element_text(size=12),
           axis_title=element_text(size=10),
           legend_text=element_text(size=8),
           legend_title=element_text(size=8),
           legend_position='top') + \
       xlab('') + \
       ylab('Importance') + coord_flip()

print(plot)

plot.save(f'{output_dir}/feature_importance.pdf', height=7, width=6)
