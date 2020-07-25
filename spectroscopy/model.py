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

from spectroscopy.utils import (
    load_training_data,
    get_wavelength_columns,
    plot_fit,
)

MODEL_DIR = Path('bin/model/')
MODEL_FILENAME = 'model.pkl'

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
    # return RandomForestRegressor(random_state=10, max_depth=20, n_estimators=100)
    # return LGBMRegressor()
    return RandomForestRegressor(random_state=10)

def extract_features(df):
    df['process_method'] = pd.get_dummies(df[['process_method']].fillna(''))
    feature_columns = get_wavelength_columns(df)
    feature_columns.append('integration_time')
    feature_columns.append('process_method')
    return feature_columns


def train_ammonia_n_model(model_dir=None):
    if model_dir is None:
        model_dir = MODEL_DIR
    model_dir = Path(model_dir)
    df = load_training_data()
    # df = df[df['process_method'] == 'ground']
    feature_columns = extract_features(df)
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
        # ('PCA', PCA()),
        ('model', model)
    ])
    model_pipeline.fit(X_train, y_train)
    selected_scores = score_model(model_pipeline, X_train, y_train, X_test, y_test)
    # print out the selected features
    feature_selector = model_pipeline.named_steps['feature_selector']
    selected_features = X_test.columns[feature_selector.get_support()]
    print('selected important features:', selected_features)
    print('normalized + reduced feature scores:')
    pprint(selected_scores)
    
    model_pipeline = Pipeline([
        ('normalizer_1', Normalizer()),
        ('feature_selector', SelectFromModel(_define_model())),
        ('normalizer_2', Normalizer()),
        # ('PCA', PCA()),
        ('model', model)
    ])
    model_pipeline.fit(X_train, y_train)
    selected_scores = score_model(model_pipeline, X_train, y_train, X_test, y_test)
    # print out the selected features
    feature_selector = model_pipeline.named_steps['feature_selector']
    selected_features = X_test.columns[feature_selector.get_support()]
    print('selected important features:', selected_features)
    print('reduced features scores:')
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


def load_model(model_dir=None):
    if model_dir is None:
        model_dir = MODEL_DIR
    model_dir = Path(model_dir)
    with open(model_dir/MODEL_FILENAME, 'rb') as f:
        model = pickle.load(f)
    return model



    