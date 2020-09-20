import logging

import matplotlib.pyplot as plt
import numpy as np
# from sklearn.impute import SimpleImputer
# TODO: write unittests


logger = logging.getLogger(__name__)

def plot_sample(wavelengths, transmittance):
    plt.figure(figsize=(20,10))
    plt.title('Transmittance vs Wavelength')
    plt.xlabel('Wavelength(nm)')
    plt.ylabel('Transmittance (counts)')
    plt.plot(wavelengths, transmittance)
    plt.show()


def get_wavelength_columns(df, lower_bound=None):
    wavelength_columns = []
    for column in df.columns:
        try:
            value = float(column)
            if lower_bound is not None:
                if value > 850:
                    wavelength_columns.append(column)
            else:
                wavelength_columns.append(column)
        except:
            pass
    return wavelength_columns
    

def plot_fit(y_true, y_pred, save=True):
    max_value = np.max(y_true)
    max_value += max_value/10
    plt.figure(figsize=(10,10))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.title('Ammonia-N Prediction from Machine Learning Spectroscopy Inference Model')
    plt.plot(np.linspace(0, max_value, len(y_true)), np.linspace(0, max_value, len(y_true)))
    plt.xlabel('True Ammonia-N')
    plt.ylabel('Predicted Ammonia-N')
    plt.xlim(0,max_value)
    plt.ylim(0,max_value)
    if save:
        plt.savefig('prediction_vs_truth.png')

def plot_residuals(y_true, y_pred, save=True):
    plt.figure(figsize=(10,10))
    plt.scatter(y_true, y_true-y_pred)
    plt.xlabel('True')
    plt.ylabel('residual')
    plt.title('Residuals')
    if save:
        plt.savefig('residuals.png')


def get_highest_error_data(y_true, model, X):
    y_pred = model.predict(X)
    residuals = y_true - y_pred
    max_error_indices = np.argmax(residuals)

# def highlight_important_wavelengths(wavelengths):
#     plt.figure(figsize=(20,10))
#     x = wavelengths
#     y = [0]*len(wavelengths)
#     plt.plot(x, y)