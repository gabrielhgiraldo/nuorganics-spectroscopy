from configparser import ConfigParser
import configparser
from pathlib import Path

import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable

import pandas as pd
# TODO: determine configuration flow
# TODO: if no configuration file is found, prompt user to input some configurations
# handle user configurations
INTERNAL_CONFIG_FILEPATH = Path(__file__).parent / 'config.ini'

def get_triggered_id():
    ctx = dash.callback_context

    if not ctx.triggered:
        return None
    else:
        return ctx.triggered[0]['prop_id'].split('.')[0]


def parse_internal_configuration():
    config_parser = ConfigParser()
    config_parser.read(INTERNAL_CONFIG_FILEPATH)


app = dash.Dash(__name__, suppress_callback_exceptions=True)
# DUMMY DATA
# data = pd.read_csv(Path(__file__).parent/'ammonia_N.csv')

# TODO: generate layout as a function
# TODO: add state handler class?
# TODO: add ability to change configurations
# TODO: add tabs
def training_data_table(data):
    return DataTable(
        id='training-data-table',
        columns=[{"name": column, "id": column} for column in data.columns],
        data=data.to_dict('records'),
        fixed_rows={'headers': True},
        style_table={
            'height': '300px',
            'overflowY': 'auto'
        }
    )

def training_uploader():
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
    )

def training_content(data):
    return html.Div([
        training_data_table(data),
        training_uploader(),
        html.Button('train model', id='train-button')
    ])

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

def settings_content():
    # TODO: get saved settings and pre-populate 
    # TODO: change input types to validate for filepath format
    return html.Div(
            id='settings-content',
            children=[
                html.Div([
                    html.Label('settings filepath'),
                    dcc.Input(
                        id='settings-filepath',
                        placeholder='e.g. ~/spectroscopy/config.ini'
                    ),
                ]),
                html.Div([
                    html.Label('training samples directory filepath'),
                    dcc.Input(
                        id='training-directory',
                        placeholder='e.g. ~/spectroscopy/data'
                    ),
                ]),
                html.Button('save settings',id='settings-save-button'),
                html.Div(id='settings-feedback')
            ]
    )
    

    

# TODO: add data state manager
# tab callback
@app.callback(
    output=Output('tab-content', 'children'),
    inputs=[Input('tabs','value')])
def render_content(tab):
    if tab == 'settings-tab':
        return settings_content()
    elif tab == 'training-tab':
        return training_content(pd.DataFrame([range(5)]*5,columns=range(5)))
    else:
        return ['no content available for tab value']

# save settings callback
@app.callback(
    output=Output('settings-feedback','children'),
    inputs=[
        Input('settings-filepath', 'value'),
        Input('training-directory', 'value'),
        Input('settings-save-button', 'n_clicks')
    ]
)
def save_settings(settings_filepath, training_directory, num_clicks):
    print('save_settings triggered', settings_filepath, training_directory, num_clicks)
    input_id = get_triggered_id()
    if input_id == 'settings-save-button':
        messages = []
        if settings_filepath is None or training_directory is None:
            if settings_filepath is None:
                messages.append(html.P('settings filepath required'))
            if training_directory is None:
                messages.append(html.P('training directory required'))
        else:
            # TODO: check if paths are valid
            messages.append('settings saved')
        return messages
        # TODO: save settings to configuration file
    else:
        raise PreventUpdate       


app.layout = render_layout
# TODO: add ability to specify data location
# TODO: add ability to retrain model(s)
# TODO: add ability to configure included model parameters
# TODO: add ability to save retrained model(s)
# TODO: add ability to view and download prediction results
if __name__ == "__main__":
    app.server.run(debug=True)