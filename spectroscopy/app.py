from configparser import SafeConfigParser
import configparser
from pathlib import Path

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable
import pandas as pd

# handle user configurations
CONFIG_FILEPATH = Path(__file__).parent / 'config.ini'
config_parser = SafeConfigParser()
config_parser.read(CONFIG_FILEPATH)
TRAINING_DIR = config_parser.get('paths', 'training_dir')
SAMPLES_DIR = config_parser.get('paths', 'samples_dir')
OUTPUT_DIR = config_parser.get('paths', 'output_dir')

app = dash.Dash(__name__)
# DUMMY DATA
# data = pd.read_csv(Path(__file__).parent/'ammonia_N.csv')


app.layout = html.Div([
    html.H3('Nuorganics Spectroscopy Modeling'),
    # DataTable(
    #     id='results-data-table',
    #     columns=[{"name": column, "id": column} for column in data.columns],
    #     data=data.to_dict('records'),
    #     fixed_rows={'headers': True},
    #     style_table={
    #         'height': '300px',
    #         'overflowY': 'auto'
    #     }
    # ),
    dcc.Upload(
        id='upload-training',
        multiple=True,
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
    ),
    html.Button('train model')
])
# TODO: add ability to specify data location
# TODO: add ability to retrain model(s)
# TODO: add ability to configure included model parameters
# TODO: add ability to save retrained model(s)
# TODO: add ability to view and download prediction results
if __name__ == "__main__":
    app.run_server(debug=True)