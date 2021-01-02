from spectroscopy.modeling.utils import WavelengthDataExtractor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def define_model():
    return Pipeline(
        steps=[
            ('preprocess', WavelengthDataExtractor()),
            ('model', RandomForestRegressor(
                # n_estimators=1000
                # max_depth=20
            )),
        ])