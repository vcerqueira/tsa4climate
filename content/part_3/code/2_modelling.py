import pandas as pd
from plotnine import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from src.tde import time_delay_embedding
from config import ASSETS, OUTPUTS

CHAPTER = 'Part 3'
assets = ASSETS[CHAPTER]
output_dir = OUTPUTS[CHAPTER]

file = f'{assets}/IrishSmartBuoy.csv'

# reading the data set
# skipping the second with skiprows
# parsing time column to datetime and setting it as index
data = pd.read_csv(file, skiprows=[1], parse_dates=['time'], index_col='time')

# defining the series and converting cm to meters
series = data['SignificantWaveHeight'] / 100
# resampling to hourly and taking the mean
series = series.resample('H').mean()

# Modeling

# using past 24 observations as explanatory variables
N_LAGS = 24
# using the next 6 hours as the forecasting horizon
HORIZON = 12
# forecasting the probability of waves above 5 meters
THRESHOLD = 5

# leaving last 20% of observations for testing
train, test = train_test_split(series, test_size=0.2, shuffle=False)

# transforming time series into a tabular format for supervised learning
X_train, Y_train = time_delay_embedding(train, n_lags=N_LAGS, horizon=HORIZON, return_Xy=True)
X_test, Y_test = time_delay_embedding(test, n_lags=N_LAGS, horizon=HORIZON, return_Xy=True)

y_train = Y_train.apply(lambda x: (x > THRESHOLD).any(), axis=1).astype(int)
y_test = Y_test.apply(lambda x: (x > THRESHOLD).any(), axis=1).astype(int)

model = RandomForestClassifier(max_depth=5)
model_lt = RandomForestClassifier(max_depth=5)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
probs_lt = model.predict_proba(X_test)[:, 1]

print(roc_auc_score(y_test, probs))

# plotting

# class distributions
class_dist = pd.concat([y_train.value_counts(normalize=True),
                        y_test.value_counts(normalize=True)], axis=0)
class_dist.index = ['Normal Waves', 'Large Waves',
                    'Normal Waves', 'Large Waves']
class_dist = class_dist.reset_index()
class_dist['Data'] = ['Train', 'Train', 'Test', 'Test']
class_dist.columns = ['Event', 'Probability', 'Data']
class_dist['Data'] = pd.Categorical(class_dist['Data'], categories=['Train', 'Test'])

dist_plt = ggplot(class_dist) + \
           aes(x='Data', y='Probability', fill='Event', group='Event') + \
           theme_classic(base_family='Palatino', base_size=12) + \
           theme(plot_margin=.15,
                 axis_text=element_text(size=11),
                 legend_title=element_blank(),
                 legend_position='top') + \
           geom_bar(stat='identity',
                    position='dodge',
                    width=0.8) + \
           xlab('') + \
           ylab('Event probability') + \
           ggtitle('') + \
           scale_fill_manual(values=['#111e6c', '#4f97a3'])
print(dist_plt)

# roc curve

fpr, tpr, thresholds = roc_curve(y_test, probs)
roc_data = pd.DataFrame({'fpr': fpr, 'tpr': tpr})

roc_plt = ggplot(roc_data) + \
          aes(x='fpr', y='tpr') + \
          theme_classic(base_family='Palatino', base_size=12) + \
          theme(plot_margin=.125,
                axis_text=element_text(size=10),
                legend_title=element_blank(),
                legend_position='top') + \
          geom_line(size=1.7, color='#007dd6') + \
          xlab('False Positive Rate') + \
          ylab('True Positive Rate') + \
          ylim(0, 1) + xlim(0, 1) + \
          ggtitle('') + \
          geom_abline(intercept=0,
                      slope=1,
                      size=1,
                      color='black',
                      linetype='dashed')

dist_plt.save(f'{output_dir}/dist_plot.pdf', height=5, width=8)
roc_plt.save(f'{output_dir}/roc_curve.pdf', height=5, width=8)
