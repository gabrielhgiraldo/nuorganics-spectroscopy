import argparse
from pathlib import Path
import os

import pandas as pd

from spectroscopy.model import load_model
from spectroscopy.utils import get_wavelength_columns, parse_trms

# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-sp", "--sample-path", required=True,
   help="path to sample files")
args = vars(ap.parse_args())

print('loading model')
model = load_model()
sample_path = Path(args['sample_path'])
print('extracting samples')
df_samples = pd.DataFrame()
try:
    df_samples = parse_trms(sample_path)
    print('samples extracted')
except FileNotFoundError:
    print(f'no samples found at the provided path {sample_path}')
    os.system.exit()
print('calculating samples Ammonia-N')
wavelength_columns = get_wavelength_columns(df_samples)
X_samples = df_samples[wavelength_columns]
ammonia_N = model.predict(X_samples)
results_df = pd.DataFrame({
    'filename':df_samples['filename'],
    'Ammonia-N':ammonia_N
})
results_path = Path('ammonia_N.csv')
print(f'saving samples Ammonia-N to {results_path}')
results_df.to_csv(results_path, index=False)

