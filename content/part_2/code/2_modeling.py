import re

import numpy as np
import pandas as pd
from plotnine import *

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error as mae, r2_score
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import RFE
from sktime.transformations.series.date import DateTimeFeatures

from config import ASSETS, OUTPUTS
from src.tde import time_delay_embedding, mts_to_tabular
from src.correlation_filter import correlation_filter
from src.holdout import Holdout

CHAPTER = 'Part 2'
assets = ASSETS[CHAPTER]
output_dir = OUTPUTS[CHAPTER]

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
mv_series.columns = [re.sub('Incoming Solar', 'Solar Irradiance', x) for x in mv_series.columns]

mv_series = mv_series.astype(float)

### train test split before cross-validation

# target variable
TARGET = 'Solar Irradiance'
# number of lags for each variable
N_LAGS = 24
# forecasting horizon for solar irradiance
HORIZON = 48

# leaving the last 30% of observations for testing
train, test = train_test_split(mv_series, test_size=0.3, shuffle=False)

# transforming the time series into a tabular format
X_train, Y_train_all = mts_to_tabular(train, N_LAGS, HORIZON, return_Xy=True)
X_test, Y_test_all = mts_to_tabular(train, N_LAGS, HORIZON, return_Xy=True)

# subsetting the target variable
target_columns = Y_train_all.columns.str.contains(TARGET)
Y_train = Y_train_all.iloc[:, target_columns]
Y_test = Y_test_all.iloc[:, target_columns]

# including datetime information to model seasonality
hourly_feats = DateTimeFeatures(ts_freq='H',
                                keep_original_columns=True,
                                feature_scope='efficient')

# building a pipeline
pipeline = Pipeline([
    # feature extraction based on datetime
    ('extraction', hourly_feats),
    # removing correlated explanatory variables
    ('correlation_filter', FunctionTransformer(func=correlation_filter)),
    # applying feature selection based on recursive feature elimination
    ('select', RFE(estimator=RandomForestRegressor(max_depth=5), step=3)),
    # building a random forest model for forecasting
    ('model', RandomForestRegressor())]
)

# parameter grid for optimization
param_grid = {
    'extraction': ['passthrough', hourly_feats],
    'select__n_features_to_select': np.linspace(start=.1, stop=1, num=10),
    'model__n_estimators': [100, 200]
}

# optimizing the pipeline with random search
model = RandomizedSearchCV(estimator=pipeline,
                           param_distributions=param_grid,
                           scoring='neg_mean_squared_error',
                           n_iter=25,
                           n_jobs=5,
                           refit=True,
                           verbose=2,
                           cv=Holdout(n=X_train.shape[0]),
                           random_state=123)

# running random search
model.fit(X_train, Y_train)

# model.cv_results_
# model.estimator
model.best_estimator_

# Pipeline(steps=[('extraction',
#                  DateTimeFeatures(feature_scope='efficient', ts_freq='H')),
#                 ('correlation_filter',
#                  FunctionTransformer(func=<function correlation_filter at 0x28cccfb50>)),
#                 ('select',
#                  RFE(estimator=RandomForestRegressor(max_depth=5),
#                      n_features_to_select=0.9, step=3)),
#                 ('model', RandomForestRegressor(n_estimators=20))])

all_feature_names = pd.Series(model.best_estimator_['select'].feature_names_in_)

selected_features = all_feature_names[model.best_estimator_['select'].support_]
importance_scores = pd.Series(dict(zip(selected_features, model.best_estimator_['model'].feature_importances_)))

imp_df = importance_scores.reset_index()
imp_df.columns = ['Feature', 'Importance']
imp_df = imp_df.sort_values('Importance', ascending=False)
imp_df['Feature'] = pd.Categorical(imp_df['Feature'], categories=imp_df['Feature'])

plot = ggplot(imp_df[:20], aes(x='Feature', y='Importance')) + \
       geom_bar(fill='#ff8b3d', stat='identity', position='dodge') + \
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

# getting forecasts for the test set
forecasts = model.predict(X_test)
forecasts = pd.DataFrame(forecasts, columns=Y_test.columns)

score = {col: r2_score(y_true=Y_test[col], y_pred=forecasts[col])
         for col in forecasts}

pd.Series(score)



###
