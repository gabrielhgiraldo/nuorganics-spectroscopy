import argparse
from pathlib import Path

from spectroscopy.model import load_model
from spectroscopy.utils import parse_trms

# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-sp", "--sample-path", required=True,
   help="path to sample files")
args = vars(ap.parse_args())

model = load_model()
sample_path = Path(args['sample-path'])

df_samples = parse_trms(sample_path)
ammonia_df = model.predict(df_samples)
ammonia_df.to_csv(f'ammonia_N.csv')

