from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


DATA_DIR = Path(__file__).parents[1] / 'data'
TRAINING_DATA_FILENAME = 'training_data.csv'

def plot_sample(wavelengths, transmittance):
    plt.figure(figsize=(20,10))
    plt.title('Transmittance vs Wavelength')
    plt.xlabel('Wavelength(nm)')
    plt.ylabel('Transmittance (counts)')
    plt.plot(wavelengths, transmittance)
    plt.show()



def _extract_spect_filename_info(filename):
    sample_name_method, _, _ = filename.partition('-')
    if '(' in sample_name_method:
        sample_name, _, process_method = sample_name_method.partition('(')
        sample_name = sample_name.strip().lower()
        process_method = process_method.strip()[:-1] # drop )
    else:
        sample_name = sample_name_method.strip().lower()
        process_method = ''
    process_method = process_method.lower()
    sample_date = pd.to_datetime(filename.partition('-')[2].partition('#')[0])
    run_number = filename.partition('#')[2].partition('.')[0]
    return sample_name, process_method, sample_date, run_number


def _extract_integration_time(extra_info):
    return extra_info.partition('->')[2]\
                     .partition('Time:')[2]\
                     .partition('Avg:')[0]\
                     .strip()\
                     .partition('ms')[0]


def parse_spect_file(path, drop_neg_values=False):
    path = Path(path)
    df = pd.read_csv(path)
    extra_info = df.columns[0]
    data = df.iloc[:,0]\
             .str.strip()\
             .str.partition(' ')
    wavelengths = data[0].astype(float)
    values = data[2].astype(float)
    if drop_neg_values:
        mask = values > 0
        values = values[mask]
        wavelengths = wavelengths[mask]
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


def parse_trm_files(directory_path=None) -> pd.DataFrame:
    if directory_path is None:
        directory_path = DATA_DIR
    directory = Path(directory_path)
    return pd.concat([parse_spect_file(filepath) for filepath in directory.glob("*.TRM")])


def parse_abs_files(directory_path=None) -> pd.DataFrame:
    if directory_path is None:
        directory_path = DATA_DIR
    directory = Path(directory_path)
    return pd.concat([parse_spect_file(filepath) for filepath in directory.glob("*.ABS")])


def _extract_lab_report_filename_info(filename):
    sample_name_date = filename.partition('-')[2]
    sample_name, _, date_extension = sample_name_date.partition('-')
    sample_name = sample_name.strip().lower()
    sample_date = pd.to_datetime(date_extension.partition('.')[0].strip())
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
    return pd.concat([parse_lab_report(lr_filepath) for lr_filepath in DATA_DIR.glob('Lab Report*.csv')])


def get_wavelength_columns(df):
    wavelength_columns = []
    for column in df.columns:
        try:
            float(column)
            wavelength_columns.append(column)
        except:
            pass
    return wavelength_columns


def cache_cleaned_data():
    df_lr = parse_lab_reports()
    df_trms = parse_trm_files()
    # fill in missing 
    df = df_trms.join(df_lr.set_index('sample_name')[['Ammonia-N']], on='sample_name')\
                                        .reset_index(drop=True)
    # df.fillna(0, inplace=True)
    df.to_csv(DATA_DIR/'training_data.csv', index=False)


def load_training_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR/TRAINING_DATA_FILENAME)


def plot_fit(y_true, y_pred, save=True):
    plt.figure(figsize=(10,10))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.title('Ammonia-N Prediction from Machine Learning Spectroscopy Inference Model')
    plt.plot(np.linspace(0, 0.6, len(y_true)), np.linspace(0, 0.6, len(y_true)))
    plt.xlabel('True Ammonia-N')
    plt.ylabel('Predicted Ammonia-N')
    plt.xlim(0,0.6)
    plt.ylim(0,0.6)
    if save:
        plt.savefig('ammonia_prediction_vs_truth.png')