import numpy as np
from spectroscopy.modeling.utils import WavelengthDataExtractor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def define_model():
    return Pipeline(
        steps=[
            ('preprocess', WavelengthDataExtractor()),
            ('model', RandomForestRegressor(
                # n_estimators=400,
                # max_features=1,
                # max_depth=5
            )),
        ])

def generate_randomsearch_grid():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {
        'model__n_estimators': n_estimators,
        'model__max_features': max_features,
        'model__max_depth': max_depth,
        'model__min_samples_split': min_samples_split,
        'model__min_samples_leaf': min_samples_leaf,
        'model__bootstrap': bootstrap
    }
    return random_grid


def generate_gridsearch_grid(randomsearch_hyperparameters=None):
    if randomsearch_hyperparameters is None:
        reference_hyperparameters = {
            'model__bootstrap': True,
            'model__max_depth': 30,
            'model__max_features': 'sqrt',
            'model__min_samples_leaf': 1,
            'model__min_samples_split': 5,
            'model__n_estimators': 400
        }
    else:
        reference_hyperparameters = randomsearch_hyperparameters

    n_estimators = reference_hyperparameters['model__n_estimators']
    max_depth = reference_hyperparameters['model__max_depth']
    min_samples_leaf = reference_hyperparameters['model__min_samples_leaf']
    min_samples_split = reference_hyperparameters['model__min_samples_split']
    return {
        'model__n_estimators':[n_estimators - 300, n_estimators - 200, n_estimators - 100, n_estimators *2],
        'model__max_depth':[max_depth + 10, max_depth + 20, max_depth + 30, max_depth + 40],
        # 'model__max_features':[]
        'model__min_samples_leaf':[min_samples_leaf, min_samples_leaf + 1, min_samples_leaf + 2],
        'model__min_samples_split':[min_samples_split - 2, min_samples_split, min_samples_split + 2]
    }
