import json
from pathlib import Path

import numpy as np
import pickle
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
# from sklearn.utils import check_arrays

from spectroscopy.utils import load_training_data, get_wavelength_columns

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
    return RandomForestRegressor(random_state=10, max_depth=5, n_estimators=10)

def train_ammonia_n_model(model_dir=None):
    if model_dir is None:
        model_dir = MODEL_DIR
    model_dir = Path(model_dir)
    df = load_training_data()
    feature_columns = get_wavelength_columns(df)
    X, y = df[feature_columns], df['Ammonia-N']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    model = _define_model()
    model.fit(X_train, y_train)
    baseline_scores = score_model(model, X_train, y_train, X_test, y_test)
    pprint(baseline_scores)
    # select k best features
    model_pipeline = Pipeline([
        ('feature_selector', SelectFromModel(_define_model())),
        ('model', model)
    ])
    model_pipeline.fit(X_train, y_train)
    selected_scores = score_model(model_pipeline, X_train, y_train, X_test, y_test)
    pprint(selected_scores)
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




    