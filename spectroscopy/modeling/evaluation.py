import numpy as np
from sklearn.metrics import mean_squared_error


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
