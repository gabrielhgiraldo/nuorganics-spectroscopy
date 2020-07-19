import json
from pathlib import Path

import numpy as np
import pickle
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from utils import load_training_data, get_wavelength_columns

MODEL_DIR = Path('bin/model/')

def score_model(model, X_train, y_train, X_test, y_test):
    return {
        'train_r2':model.score(X_train, y_train),
        'train_rms3':np.sqrt(mean_squared_error(y_train, model.predict(X_train))),
        'test_r2':model.score(X_test, y_test),
        'test_rmse':np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    }

def train_ammonia_n_model(model_dir=None):
    if model_dir is None:
        model_dir = MODEL_DIR
    model_dir = Path(model_dir)
    df = load_training_data()
    feature_columns = get_wavelength_columns(df)
    X, y = df[feature_columns], df['Ammonia-N']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    model = RandomForestRegressor(random_state=10, max_depth=5, n_estimators=10)
    # select k best features
    model.fit(X_train, y_train)
    feature_selector = SelectFromModel(model, prefit=True)
    X_train_selected = feature_selector.transform(X_train)
    baseline_scores = score_model(model, X_train, y_train, X_test, y_test)
    pprint(baseline_scores)
    selected_features = X_test.columns[feature_selector.get_support()]
    X_test_selected = X_test[selected_features]
    model.fit(X_train_selected, y_train)
    selected_scores = score_model(model, X_train_selected, y_train, X_test_selected, y_test)
    pprint(selected_scores)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR/'baseline_scores.json', 'w') as f:
        json.dump(baseline_scores, f)
    with open(MODEL_DIR/'selected_scores.json', 'w') as f:
        json.dump(selected_scores, f)
    with open(MODEL_DIR / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_ammonia_n_model()



    