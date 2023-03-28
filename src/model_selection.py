import copy

from sklearn.base import BaseEstimator
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor

from src.lag_summary_stats import feature_engineering


class MetaEstimator(BaseEstimator):
    pass


METHOD_PARAMETERS = {
    'Ridge': {
        'model': [RidgeCV()],
        'model__alphas': [1, .5, .25, .75]
    },
    'RandomForest': {
        'model': [RandomForestRegressor()],
        'model__n_estimators': [10, 20],
        'model__max_depth': [2, 3],
    },
}

FEATURE_EXTRACTION_FUNCS = {
    'None': None,
    'basic': feature_engineering,
}

models = [*METHOD_PARAMETERS]
search_space = [METHOD_PARAMETERS[mod] for mod in models]

feature_funcs = [*FEATURE_EXTRACTION_FUNCS.values()]

search_space_with_feature_ext = copy.deepcopy(search_space)
for i, conf in enumerate(search_space_with_feature_ext):
    search_space_with_feature_ext[i]['feature__func'] = feature_funcs
