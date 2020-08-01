import json
from pathlib import Path

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
import pickle
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from skorch.regressor import NeuralNetRegressor
from torch import nn
import torch.nn.functional as F


from spectroscopy.utils import (
    load_training_data,
    get_wavelength_columns,
    plot_fit,
)

MODEL_DIR = Path('bin/model/')
MODEL_FILENAME = 'model.pkl'
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
        'train_rms3':np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_r2':model.score(X_test, y_test),
        'test_mape':mean_absolute_percentage_error(y_test, y_test_pred),
        'test_rmse':np.sqrt(mean_squared_error(y_test, y_test_pred))
    }

def _define_model():
#     return NeuralNetRegressor(
#         RegressorModule,
#         max_epochs=20,
#         lr=0.1,
# #     device='cuda',  # uncomment this to train with CUDA
#     )
    # return RandomForestRegressor(random_state=10, max_depth=20, n_estimators=100)
    # return LGBMRegressor()
    return RandomForestRegressor(random_state=10)    

# TODO: get ammonia features vs get moisture features
def get_features(df):
    feature_columns = get_wavelength_columns(df)
    feature_columns.append('integration_time')
    feature_columns.extend(df['process_method'].unique())
 
    return feature_columns


def train_ammonia_n_model(model_dir=None):
    if model_dir is None:
        model_dir = MODEL_DIR
    model_dir = Path(model_dir)
    df = load_training_data()
    # TODO: have preprocessing and feature extraction occur in model pipeline
    df['process_method'].fillna('none', inplace=True)
    df['process_method'] = df['process_method'].astype(str)
    # df = df[df['process_method'].isin(['ground','wet'])]
    df = pd.concat([df, pd.get_dummies(df['process_method'])], axis=1)
    feature_columns = get_features(df)
    X, y = df[feature_columns], df['Ammonia-N']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    print(f'total samples:{len(df)} training on:{len(X_train)} testing on {len(X_test)}')
    model = _define_model()
    model.fit(X_train, y_train)
    baseline_scores = score_model(model, X_train, y_train, X_test, y_test)
    print('all features scores:')
    pprint(baseline_scores)

    # select k best features
    model_pipeline = Pipeline([
        ('feature_selector', SelectFromModel(_define_model())),
        ('model', model)
    ])
    model_pipeline.fit(X_train, y_train)
    selected_scores = score_model(model_pipeline, X_train, y_train, X_test, y_test)
    feature_selector = model_pipeline.named_steps['feature_selector']
    selected_features = X_test.columns[feature_selector.get_support()]
    print('selected important features:', selected_features)
    print('reduced feature scores:')
    pprint(selected_scores)

  
    print('saving fit graph')
    plot_fit(y_test, model_pipeline.predict(X_test))
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR/'baseline_scores.json', 'w') as f:
        json.dump(baseline_scores, f)
    with open(MODEL_DIR/'selected_scores.json', 'w') as f:
        json.dump(selected_scores, f)
    with open(MODEL_DIR / MODEL_FILENAME, 'wb') as f:
        pickle.dump(model_pipeline, f)

# TODO: load ammonia model vs load moisture model
def load_model(model_dir=None):
    if model_dir is None:
        model_dir = MODEL_DIR
    model_dir = Path(model_dir)
    with open(model_dir/MODEL_FILENAME, 'rb') as f:
        model = pickle.load(f)
    return model



    