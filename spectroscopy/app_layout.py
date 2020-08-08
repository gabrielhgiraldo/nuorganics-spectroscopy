from pathlib import Path

import pandas as pd

from spectroscopy.utils import get_wavelength_columns
from spectroscopy.app_utils import get_user_settings, load_training_data
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable

# TODO: fix certain columns on the left
# TODO: order columns

def training_data_table():
    try:
        data = load_training_data()
        data = data.drop(get_wavelength_columns(data), axis=1)
    except FileNotFoundError:
        data = pd.DataFrame()
    return DataTable(
        id='training-data-table',
        columns=[{"name": column, "id": column} for column in data.columns],
        data=data.to_dict('records'),
        fixed_rows={'headers': True},
        style_table={
            'height': '800px',
            'overflowY': 'auto',
            'overflowX':'auto'
        },
        style_cell={
            'height': 'auto',
            # all three widths are needed
            'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
            'whiteSpace': 'normal'
        }    
    )

def training_uploader():
    return dcc.Upload(
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
    )


# TODO: adjust column ordering in training datatable
def training_content():
    return html.Div([
        training_uploader(),
        training_data_table(),
        html.Button('train model', id='train-button')
    ])

    
def setting_inputs():
    user_settings = get_user_settings()
    inputs = []
    for _, section in user_settings.items():
        for setting, value in section.items():
            inputs.append(html.Div(
                children=[
                    html.Label(setting),
                    dcc.Input(
                        id=setting,
                        value=value,
                        placeholder=f'e.g. {value}',
                        style={'width':'100%'}
                    )
                ],
                style={'padding':'10px'}
            ))
    return inputs
    
def settings_content():
    # TODO: change input types to validate for filepath format
    return html.Div(
        id='settings-content',
        children=[
            *setting_inputs(),
            html.Button('save settings',id='save-settings', n_clicks=0),
            html.Div(id='settings-feedback')
        ]
    )

def render_layout():
    return html.Div([
        html.H3('Nuorganics Spectroscopy Modeling'),
        dcc.Tabs(
            id='tabs',
            value='training-tab',
            children=[
                dcc.Tab(label='Settings', value='settings-tab'),
                dcc.Tab(label='Training', value='training-tab'),
                dcc.Tab(label='Inference', value='inference-tab'),
                dcc.Tab(label='Analysis', value='analysis-tab')
            ]),
        html.Div(id='tab-content')
    ])


