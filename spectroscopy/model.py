import json
import logging
from pathlib import Path

# from lightgbm import LGBMRegressor
import numpy as np
import pickle
from pprint import pprint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
# from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
# from sklearn.decomposition import PCA
# from skorch.regressor import NeuralNetRegressor
from torch import nn
import torch.nn.functional as F


from spectroscopy.utils import (
    TRAINING_DATA_FILENAME,
    load_extracted_data,
    get_wavelength_columns,
    plot_fit,
)
MODEL_DIR = Path('bin/model/')
MODEL_FILENAME = 'model.pkl'
MODEL_METRICS_FILENAME = 'scores.json'

logger = logging.getLogger(__name__)

class RegressorModule(nn.Module):
    def __init__(
            self,
            num_units=10,
            nonlin=F.relu,
    ):
        super(RegressorModule, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = F.relu(self.dense1(X))
        X = self.output(X)
        return X

def mean_absolute_percentage_error(y_true, y_pred): 
    # y_true, y_pred = check_arrays(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def score_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return {
        'train_r2':model.score(X_train, y_train),
        'train_mape':mean_absolute_percentage_error(y_train, y_train_pred),
        'train_rmse':np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_r2':model.score(X_test, y_test),
        'test_mape':mean_absolute_percentage_error(y_test, y_test_pred),
        'test_rmse':np.sqrt(mean_squared_error(y_test, y_test_pred))
    }

def define_model():
#     return NeuralNetRegressor(
#         RegressorModule,
#         max_epochs=20,
#         lr=0.1,
# #     device='cuda',  # uncomment this to train with CUDA
#     )
    # return RandomForestRegressor(random_state=10, max_depth=20, n_estimators=100)
    # return LGBMRegressor()
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(), ['process_method'])
        ],
        remainder='passthrough'
    )
    return Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', RandomForestRegressor(random_state=10))
    ])

def get_features(df):
    feature_columns = get_wavelength_columns(df)
    feature_columns.append('integration_time')
    feature_columns.append('process_method')
    return feature_columns

# TODO: make this an sklearn transformer or series of transformers
# TODO: enable different features for different models
def transform_data(df):
    df['process_method'].fillna('none', inplace=True)
    df['process_method'] = df['process_method'].astype(str)
    # df = df[df['process_method'].isin(['ground','wet'])]
    # df = pd.concat([df, pd.get_dummies(df['process_method'])], axis=1)
    feature_columns = get_features(df)
    X = df[feature_columns]
    return X


# TODO: generalize this for multiple targets
def train_models(targets, model_dir=None, training_data_path=None):
    if model_dir is None:
        model_dir = MODEL_DIR
    model_dir = Path(model_dir)
    df = load_extracted_data(training_data_path, TRAINING_DATA_FILENAME)
    # TODO: make this an sklearn transformer
    X = transform_data(df)
    # TODO: have feature extraction occur in model pipeline
    # TODO: switch printing with logging
    # TODO: add ability to include different experiments in one training run
    for target in targets:
        logger.info(f'Fitting {target} model')
        target_model_dir = model_dir / target
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
        logger.info(f'total samples:{len(df)} training on:{len(X_train)} testing on {len(X_test)}')
        # TODO: allow for different architectures for each model
        model = define_model()
        model.fit(X_train, y_train)
        scores = score_model(model, X_train, y_train, X_test, y_test)
        logger.info(pprint(scores))
        logger.info('saving fit graph')
        plot_fit(y_test, model.predict(X_test))
        target_model_dir.mkdir(parents=True, exist_ok=True)
        # TODO: use database or experiment handling framework for metrics storage
        with open(target_model_dir/MODEL_METRICS_FILENAME, 'w') as f:
            json.dump(scores, f)
        with open(target_model_dir / MODEL_FILENAME, 'wb') as f:
            pickle.dump(model, f)


def load_model(model_target, model_dir=None):
    if model_dir is None:
        model_dir = MODEL_DIR
    model_dir = Path(model_dir) / model_target
    with open(model_dir/MODEL_FILENAME, 'rb') as f:
        model = pickle.load(f)
    return model


def load_model_metrics(model_target, model_dir=None):
    if model_dir is None:
        model_dir = MODEL_DIR
    model_dir = Path(model_dir) / model_target
    with open(model_dir / MODEL_METRICS_FILENAME) as f:
        scores = json.load(f)
    return scores