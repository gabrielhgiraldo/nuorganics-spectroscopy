from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.impute import SimpleImputer


DATA_DIR = Path(__file__).parent / 'data'
TRAINING_DATA_FILENAME = 'training_data.csv'

def plot_sample(wavelengths, transmittance):
    plt.figure(figsize=(20,10))
    plt.title('Transmittance vs Wavelength')
    plt.xlabel('Wavelength(nm)')
    plt.ylabel('Transmittance (counts)')
    plt.plot(wavelengths, transmittance)
    plt.show()


def _extract_trm_filename_info(filename):
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


def parse_trm(trm_path, drop_neg_trans=False):
    trm_path = Path(trm_path)
    df = pd.read_csv(trm_path)
    extra_info = df.columns[0]
    data = df.iloc[:,0]\
             .str.strip()\
             .str.partition(' ')
    wavelengths = data[0].astype(float)
    transmittance = data[2].astype(float)
    if drop_neg_trans:
        mask = transmittance > 0
        transmittance = transmittance[mask]
        wavelengths = wavelengths[mask]
    sample_df = pd.DataFrame([transmittance.values], columns=wavelengths)
    sample_df['extra_info'] = extra_info
    sample_df['filename'] = trm_path.name
    sample_name, process_method, sample_date, run_number = _extract_trm_filename_info(trm_path.name)
    sample_df['sample_name'] = sample_name
    sample_df['sample_date'] = sample_date
    sample_df['run_number'] = run_number
    sample_df['process_method'] = process_method
    sample_df = sample_df.reset_index(drop=True)
    return sample_df

def parse_trms(trm_directory=None) -> pd.DataFrame:
    if trm_directory is None:
        trm_directory = DATA_DIR
    trm_directory = Path(trm_directory)
    return pd.concat([parse_trm(trm_filepath) for trm_filepath in trm_directory.glob("*.TRM")])

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
    df_trms = parse_trms()
    # fill in missing 
    df = df_trms.join(df_lr.set_index('sample_name')[['Ammonia-N']], on='sample_name')\
                                        .reset_index(drop=True)
    # df.fillna(0, inplace=True)
    df.to_csv(DATA_DIR/'training_data.csv', index=False)


def load_training_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR/TRAINING_DATA_FILENAME)
