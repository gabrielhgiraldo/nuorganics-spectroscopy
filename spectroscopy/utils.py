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


def get_wavelength_columns(df, lower_bound=850, upper_bound=1625):
    wavelength_columns = []
    for column in df.columns:
        try:
            value = float(column)
            if lower_bound is not None and value < lower_bound \
                or upper_bound is not None and value > upper_bound:
                pass
            else:
                wavelength_columns.append(column)
        except:
            pass
    return wavelength_columns
    

def plot_pred_v_actual(y_true, y_pred, save=True, save_dir=None, model_target=None,
                            save_filename='pred_v_actual.png'):
    max_value = np.max(y_true)
    max_value += max_value/10
    fig = plt.figure(figsize=(10,10))
    plt.scatter(y_true, y_pred, alpha=0.5)
    # add 'ideal fit' line
    plt.plot(np.linspace(0, max_value, len(y_true)), np.linspace(0, max_value, len(y_true)))
    # add title, labels, and limits
    plt.title(f'{model_target} predicted vs actual')
    plt.xlabel(f'True {model_target}')
    plt.ylabel(f'Predicted {model_target}')
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)
    if save:
        if save_dir is not None:
            save_path = save_dir/save_filename
        else:
            save_path = save_filename
        fig.savefig(save_path)
        return fig, save_path
    return fig


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