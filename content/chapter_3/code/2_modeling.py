import numpy as np
import pandas as pd
from plotnine import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.tde import time_delay_embedding
from src.model_selection import (MetaEstimator,
                                 search_space_with_feature_ext)


from config import CHAPTER_ASSETS, CHAPTER_OUTPUTS

CHAPTER = 'Chapter 3'
assets = CHAPTER_ASSETS[CHAPTER]
output_dir = CHAPTER_OUTPUTS[CHAPTER]

data = pd.read_csv(f'{assets}/wind_df.csv', parse_dates=['datetime'], index_col='datetime')
data.drop('rec_fcast', axis=1, inplace=True)
data = data.loc[[2013 < x < 2018 for x in data.index.year], :]
data.columns = ['Wind Power', 'Inst. Capacity']


data['Norm. Wind Power'] = data['Wind Power'] / data['Inst. Capacity']

series = data['Norm. Wind Power'].resample('H').mean()
series.iloc[np.where(series > 1)] = np.nan
series = series.ffill()

# modeling

# leaving last 20% of observations for testing
train, test = train_test_split(series, test_size=0.2, shuffle=False)

# transforming time series into a tabular format for supervised learning
X_train, Y_train = time_delay_embedding(train, n_lags=24, horizon=24, return_Xy=True)
X_test, Y_test = time_delay_embedding(test, n_lags=24, horizon=24, return_Xy=True)



# print(X_train.head().round(2).to_latex())

# Create a pipeline for hyperparameter optimization
# 'feature' contains different possibilities for feature extraction
# 'model' contains different regression algorithms and respective hyperparameters
pipeline = Pipeline([('feature', FunctionTransformer()),
                     ('model', MetaEstimator())])

# do random search optimization for model selection
search_mod = RandomizedSearchCV(estimator=pipeline,
                                param_distributions=search_space_with_feature_ext,
                                scoring='r2',
                                n_iter=30,
                                n_jobs=1,
                                refit=True,
                                verbose=2,
                                cv=TimeSeriesSplit(n_splits=3),
                                random_state=123)

search_mod.fit(X_train.head(1000), Y_train.head(1000))

print(search_mod.best_estimator_)

# forecasting testing observations using the selected model
Y_hat_test = search_mod.predict(X_test)
Y_hat_test = pd.DataFrame(Y_hat_test, columns=Y_train.columns)

# evaluating the selected model over the forecasting horizon
r2_scores = {col: r2_score(y_true=Y_test[col],
                           y_pred=Y_hat_test[col])
             for col in Y_hat_test}

### performance by horizon

results = pd.Series(r2_scores).reset_index()
results.columns = ['Horizon', 'Score']
results['Horizon'] = [s[s.find("(") + 1:s.find(")")] for s in results['Horizon']]
results['Horizon'] = pd.Categorical(results['Horizon'], categories=results['Horizon'])
results['alpha'] = results['Score'] + .5

plt_results = ggplot(results) + \
              aes(x='Horizon', y='Score', fill='Score') + \
              theme_538(base_family='Palatino', base_size=12) + \
              theme(plot_margin=.2,
                    axis_text_y=element_text(size=12),
                    axis_text_x=element_text(size=8)) + \
              geom_bar(stat='identity',
                       position='dodge') + \
              xlab('') + \
              ylab('R2 Score') + \
              scale_fill_gradient2(low='#e8ffff',
                                   mid='#6495ed',
                                   high='#1e3f66') + \
              guides(fill=False)

### performance on dist

target_h = 'Norm. Wind Power(t+3)'
df = pd.DataFrame({'Predicted': Y_hat_test[target_h].values,
                   'Actual': Y_test[target_h].values}, )

plt_dist = ggplot(df) + \
           aes(x='Predicted', y='Actual') + \
           theme_538(base_family='Palatino', base_size=12) + \
           theme(plot_margin=.175,
                 axis_text_y=element_text(size=10),
                 axis_text_x=element_text(size=10)) + \
           geom_point(color='#192841') + \
           xlim(0, .8) + \
           ylim(0, .8) + \
           geom_abline(intercept=0,
                       slope=1,
                       size=1.2,
                       color='red',
                       linetype='dashed') + \
           xlab('Predicted values') + \
           ylab('Actual values')

plt_results.save(f'{output_dir}/wind_results.pdf', height=5, width=13)
plt_dist.save(f'{output_dir}/wind_results_extreme.pdf', height=7, width=8)
