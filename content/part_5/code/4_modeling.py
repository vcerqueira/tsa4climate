import pandas as pd

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from plotnine import *

from src.lgbm_optuna import optimize_params

from config import ASSETS, OUTPUTS

PART = 'Part 5'
assets = ASSETS[PART]
output_dir = OUTPUTS[PART]

hail_df = pd.read_csv(f'{assets}/storms_processed.csv')

hail_df['target'].value_counts(normalize=True)

hail_df.isna().mean()
hail_df.groupby('target').mean()

hail_df = hail_df.loc[:, hail_df.isna().mean() < 0.2]
hail_df = hail_df.dropna()

hail_df['target'].value_counts(normalize=True)

# Modeling


X, y = hail_df.drop('target', axis=1), hail_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

params = optimize_params(X_train, y_train, n_trials=100)

dtrain = lgb.Dataset(X_train, label=y_train)

gbm = lgb.train(params, dtrain)
preds = gbm.predict(X_test)
preds = (preds - preds.min() / preds.max() - preds.min())
# preds[preds < 0] = 0

# print(brier_score_loss(y_test, preds))
print(roc_auc_score(y_test, preds))

fpr, tpr, thresholds = roc_curve(y_test, preds)
roc_data = pd.DataFrame({'fpr': fpr, 'tpr': tpr})

roc_plt = ggplot(roc_data) + \
          aes(x='fpr', y='tpr') + \
          theme_classic(base_family='Palatino', base_size=12) + \
          theme(plot_margin=.125,
                axis_text=element_text(size=10),
                legend_title=element_blank(),
                legend_position='top') + \
          geom_line(size=1.7, color='#4f97a3') + \
          xlab('False Positive Rate') + \
          ylab('True Positive Rate') + \
          ylim(0, 1) + xlim(0, 1) + \
          ggtitle('') + \
          geom_abline(intercept=0,
                      slope=1,
                      size=1,
                      color='black',
                      linetype='dashed')

roc_plt.save(f'{output_dir}/roc_curve.pdf', height=5, width=8)
