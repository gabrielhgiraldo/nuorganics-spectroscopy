import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from sklearn.impute import SimpleImputer

SCAN_FILE_DATETIME_FORMAT = '%m-%d-%y'
DATA_DIR = Path(__file__).parents[1] / 'data'
TRAINING_DATA_FILENAME = 'training_data.csv'
INFERENCE_RESULTS_FILENAME = 'inference_results.csv'

AMMONIA_N = 'Ammonia-N'
TOTAL_N = 'N'
PERCENT_MOISTURE = 'Moisture'
PHOSPHORUS = 'P'
POTASSIUM = 'K'
SULFUR = 'S'
AVAILABLE_TARGETS = [
    AMMONIA_N,
    TOTAL_N,
    PERCENT_MOISTURE,
    PHOSPHORUS,
    POTASSIUM,
    SULFUR
]

logger = logging.getLogger(__name__)

def plot_sample(wavelengths, transmittance):
    plt.figure(figsize=(20,10))
    plt.title('Transmittance vs Wavelength')
    plt.xlabel('Wavelength(nm)')
    plt.ylabel('Transmittance (counts)')
    plt.plot(wavelengths, transmittance)
    plt.show()



def _extract_spect_filename_info(filename):
    sample_name_method, _, remaining = filename.partition('-')
    if '(' in sample_name_method:
        sample_name, _, process_method = sample_name_method.partition('(')
        sample_name = sample_name.strip().lower()
        process_method = process_method.strip()[:-1] # drop )
    else:
        sample_name = sample_name_method.strip().lower()
        process_method = ''
    process_method = process_method.lower()
    sample_date_string = re.search(r'\d+-\d+-\d+',remaining)[0].strip()
    sample_date = pd.to_datetime(sample_date_string, format=SCAN_FILE_DATETIME_FORMAT)
    run_number = filename.partition('#')[2].partition('.')[0]
    return sample_name, process_method, sample_date, run_number


def _extract_integration_time(extra_info):
    return extra_info.partition('->')[2]\
                     .partition('Time:')[2]\
                     .partition('Avg:')[0]\
                     .strip()\
                     .partition('ms')[0]


def parse_spect_file(path):
    path = Path(path)
    df = pd.read_csv(path)
    extra_info = df.columns[0]
    data = df.iloc[:,0]\
             .str.strip()\
             .str.partition(' ')
    wavelengths = data[0].astype(float)
    values = data[2].astype(float)
    sample_df = pd.DataFrame([values.values], columns=wavelengths)
    sample_df['extra_info'] = extra_info
    sample_df['integration_time'] = _extract_integration_time(extra_info)
    sample_df['filename'] = path.name
    sample_name, process_method, sample_date, run_number = _extract_spect_filename_info(path.name)
    sample_df['sample_name'] = sample_name
    sample_df['sample_date'] = sample_date
    sample_df['run_number'] = run_number
    sample_df['process_method'] = process_method
    sample_df = sample_df.reset_index(drop=True)
    return sample_df


def parse_trm_files(directory_path=None, zero_negatives=True) -> pd.DataFrame:
    if directory_path is None:
        directory_path = DATA_DIR
    directory_path = Path(directory_path)
    try:
        trm_filepaths = directory_path.glob("*.TRM")
        with ThreadPoolExecutor(max_workers=5) as pool:
            dfs = pool.map(parse_spect_file, trm_filepaths)
        df_trms = pd.concat(dfs)
        if zero_negatives:
            # set trms that are < 0 to 0
            num = df_trms._get_numeric_data()
            num[num < 0] = 0
        return df_trms
    except ValueError:
        raise FileNotFoundError(f'no .TRM files found at {directory_path}')


def parse_abs_files(directory_path=None) -> pd.DataFrame:
    if directory_path is None:
        directory_path = DATA_DIR
    directory_path = Path(directory_path)
    try:
        return pd.concat([parse_spect_file(filepath) for filepath in directory_path.glob("*.ABS")])
    except ValueError:
        raise FileNotFoundError(f'no .ABS files found at {directory_path}')

def _extract_lab_report_filename_info(filename):
    sample_name_date = filename.partition('-')[2]
    sample_name = re.split(r'\d+-\d+-\d+', sample_name_date)[0]\
                     .partition('-')[0]\
                     .strip()
    sample_name = sample_name.strip().lower()
    sample_date_string = re.search(r'\d+-\d+-\d+',sample_name_date)[0].strip()
    sample_date = pd.to_datetime(sample_date_string, format=SCAN_FILE_DATETIME_FORMAT)
    return sample_name, sample_date

def parse_lab_report(filepath) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['filename'] = filepath.name
    sample_name, sample_date = _extract_lab_report_filename_info(filepath.name)
    df['sample_name'] = sample_name
    df['sample_date'] = sample_date
    df = df.reset_index(drop=True)
    return df

def parse_lab_reports(lab_report_directory=None) -> pd.DataFrame:
    if lab_report_directory is None:
        lab_report_directory = DATA_DIR
    lab_report_directory = Path(lab_report_directory)

    try:
        lr_filepaths = lab_report_directory.glob('Lab Report*.csv')
        with ThreadPoolExecutor(max_workers=5) as pool:
            reports = pool.map(parse_lab_report, lr_filepaths)
        return pd.concat(reports)
    except ValueError:
        raise FileNotFoundError(f'no lab report files found at {lab_report_directory}')

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

# TODO: incorporate sample date
def check_data_sample_name_match(df_lr, df_samples):
    unmatched_names = set(df_lr['sample_name'].unique()) - set(df_samples['sample_name'].unique())
    return unmatched_names

# TODO: improve performance of extraction with concurrency
def extract_data(data_path=DATA_DIR, extracted_filename=TRAINING_DATA_FILENAME, cache=True):
    # handle transmittance
    df_trms = parse_trm_files(data_path)
    # try and see if there's groundtruth for these files
    try:
        df_lr = parse_lab_reports(data_path)
    except FileNotFoundError as e:
        logger.warning(e)
        df = df_trms
    else:
        # if there's groundtruth, join the groundtruth to the dataset
        unmatched_names = check_data_sample_name_match(df_lr, df_trms)
        # TODO: make this a warning
        if len(unmatched_names) > 0:
            print(f'unable to match sample lab reports named {unmatched_names}')
        lab_report_columns = ['filename', *AVAILABLE_TARGETS]
        lr_to_join = df_lr.set_index(['sample_name', 'sample_date'])[lab_report_columns]
        df = df_trms.join(lr_to_join, on=['sample_name', 'sample_date'], lsuffix='_trm', rsuffix='_lr')\
                                            .reset_index(drop=True)
        # drop null Ammonia-N (unmatched)
        df = df.dropna(subset=AVAILABLE_TARGETS)
    if cache:
        df.to_csv(data_path/extracted_filename, index=False)
    return df

def load_extracted_data(directory=DATA_DIR, filename=TRAINING_DATA_FILENAME) -> pd.DataFrame:
    path = directory / filename
    return pd.read_csv(path)
# alias for load_extracted_data
load_training_data = load_extracted_data


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