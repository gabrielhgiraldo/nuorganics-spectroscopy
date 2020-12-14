import json                 
import logging
from pathlib import Path
# from spectroscopy.app_layout import predicted_vs_actual_graph

# from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
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
# from torch import nn
# import torch.nn.functional as F


from spectroscopy.data import (
    EXTRACTED_REFERENCE_FILENAME,
    get_sample_ids,
    load_cached_extracted_data,
    AVAILABLE_TARGETS
)

from spectroscopy.utils import(
    get_wavelength_columns,
    plot_pred_v_actual,
)
MODEL_DIR = Path('bin/model/')
MODEL_FILENAME = 'model.pkl'
MODEL_METRICS_FILENAME = 'scores.json'
MODEL_PRED_ACT_GRAPH_FILENAME = 'pred_v_actual.png'
MODEL_DATA_DICT_FILENAME = 'data_dict.pkl'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# class RegressorModule(nn.Module):
#     def __init__(
#             self,
#             num_units=10,
#             nonlin=F.relu,
#     ):
#         super(RegressorModule, self).__init__()
#         self.num_units = num_units
#         self.nonlin = nonlin

#         self.dense0 = nn.Linear(20, num_units)
#         self.nonlin = nonlin
#         self.dense1 = nn.Linear(num_units, 10)
#         self.output = nn.Linear(10, 1)

#     def forward(self, X, **kwargs):
#         X = self.nonlin(self.dense0(X))
#         X = F.relu(self.dense1(X))
#         X = self.output(X)
#         return X

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
        'train':{
            'r2':model.score(X_train, y_train),
            'mape':mean_absolute_percentage_error(y_train, y_train_pred),
            'rmse':np.sqrt(mean_squared_error(y_train, y_train_pred)),
        },
        'test':{
            'r2':model.score(X_test, y_test),
            'mape':mean_absolute_percentage_error(y_test, y_test_pred),
            'rmse':np.sqrt(mean_squared_error(y_test, y_test_pred))
        }
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
            ('categorical', OneHotEncoder(handle_unknown='ignore'), ['process_method'])
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


def train_models(targets=AVAILABLE_TARGETS, data=None, model_dir=None, training_data_path=None,
                 evaluate=True):
    if model_dir is None:
        model_dir = MODEL_DIR
    model_dir = Path(model_dir)
    if data is None:
        # TODO: exctract data if there's no cached
        data = load_cached_extracted_data(EXTRACTED_REFERENCE_FILENAME, training_data_path)
    # TODO: make this an sklearn transformer
    X = transform_data(data)
    # TODO: have feature extraction occur in model pipeline
    # TODO: add ability to include different experiments in one training run
    artifacts= {
        'metrics':{},
        'graphs':{},
        'data':{}
    }
    models = {}
    for target in targets:
        logger.info(f'Fitting {target} model')
        target_model_dir = model_dir / target
        y = data[target]
        # drop the samples that are missing this target
        mask = y.isna()
        y_temp = y[~mask]
        X_temp = X[~mask]
        X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=10)
        logger.info(f'total samples:{len(data)} training on:{len(X_train)} testing on {len(X_test)}')
        # TODO: allow for different architectures for each model
        model = define_model()
        # remove the index
        model.fit(X_train, y_train)
        models[target] = model
        # save model
        target_model_dir.mkdir(parents=True, exist_ok=True)
        with open(target_model_dir / MODEL_FILENAME, 'wb') as f:
            pickle.dump(model, f)

        if evaluate:
            scores = score_model(model, X_train, y_train, X_test, y_test)
            logger.info(pprint(scores))
            logger.info('saving fit graph')
            y_pred = pd.Series(model.predict(X_test), index=X_test.index)
            # create predicted vs actual graph & save img version
            _, fig_save_path = plot_pred_v_actual(
                model_target=target,
                y_true=y_test,
                y_pred=y_pred,
                save_dir = target_model_dir
            )
            # TODO: use database or experiment handling framework for metrics storage
            # save metrics
            with open(target_model_dir/MODEL_METRICS_FILENAME, 'w') as f:
                json.dump(scores, f)
            # get data associated with index
            test_samples = data[data.index.isin(X_test.index)]
            # save data
            data_dict = {
                'y_pred': y_pred,
                'y_test': y_test, 
                'test_samples': test_samples,
            }
            with open(target_model_dir/MODEL_DATA_DICT_FILENAME, 'wb') as f:
                pickle.dump(data_dict, f)
            artifacts['metrics'][target] = scores
            artifacts['graphs'][target] = [fig_save_path]
            artifacts['data'][target] = data_dict
    if evaluate:
        return artifacts, models
    return models


def load_model(model_target, model_dir=None):
    if model_dir is None:
        model_dir = MODEL_DIR
    model_dir = Path(model_dir) / model_target
    with open(model_dir/MODEL_FILENAME, 'rb') as f:
        model = pickle.load(f)
    return model


def load_model_metrics(model_target, model_dir=None):
    """Load performance metrics stored for particular target from corresponding folder"""
    if model_dir is None:
        model_dir = MODEL_DIR
    model_dir = Path(model_dir) / model_target
    with open(model_dir / MODEL_METRICS_FILENAME) as f:
        scores = json.load(f)
    return scores


def get_model_graph_paths(model_target, model_dir=None):
    # get all graphs in the path 
    model_dir = Path(model_dir) / model_target
    graph_paths = list(model_dir.glob('*.png'))
    return graph_paths
# TODO: do we want to generate the graphs once and display them as images on the UI?
# TODO: OR do we want to predict on the test data and generate the graphs dynamically?
# def load_model_graphs(model_target, model_dir=None):
#     if model_dir is None:
#         model_dir = MODEL_DIR
#     model_dir = Path(model_dir) / model_target
#     with open(model_dir / MODEL_PRED_ACT_GRAPH_FILENAME) as f:
#         # pred_v_actual_graph = predicted_vs_actual_graph()
#     return scores


def load_model_data(target, model_dir):
    with open(model_dir/target/MODEL_DATA_DICT_FILENAME, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict



def load_all_performance_artifacts(model_dir=None):
    """load all performance artifacts in the model directory (metrics, graphs, etc.)"""
    artifacts = {
        'metrics':{},
        'graphs':{},
        'data':{}
    }
    for target in AVAILABLE_TARGETS:
        try:
            model_metrics = load_model_metrics(target, model_dir)
            # get saved model graph paths
            # model_graphs = load_model_graphs(target, model_dir)
            model_graph_paths = get_model_graph_paths(target, model_dir)
            model_data_dict = load_model_data(target, model_dir)
            
        except FileNotFoundError as e:
            logger.warning(e)
        else:
            artifacts['metrics'][target] = model_metrics
            artifacts['graphs'][target] = model_graph_paths
            artifacts['data'][target] = model_data_dict
    return artifacts