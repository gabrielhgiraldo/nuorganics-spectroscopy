import argparse
from pathlib import Path
import os

import pandas as pd

from spectroscopy.model_utils import load_model, get_features
from spectroscopy.utils import parse_trm_files

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
    df_samples = parse_trm_files(sample_path)
    print('samples extracted')
except FileNotFoundError:
    print(f'no samples found at the provided path {sample_path}')
    os.system.exit()
# TODO have feature extraction happen in model pipeline
print('calculating samples Ammonia-N')
feature_columns = get_features(df_samples)
X_samples = df_samples[feature_columns]
ammonia_N = model.predict(X_samples)
results_df = pd.DataFrame({
    'sample_name': df_samples['sample_name'],
    'trm_filename':df_samples['filename'],
    'Ammonia-N':ammonia_N
})
results_path = Path('ammonia_N.csv')
print(f'saving samples Ammonia-N to {results_path}')
results_df.to_csv(results_path, index=False)

