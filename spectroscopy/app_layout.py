from spectroscopy.app_utils import get_user_settings
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable

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

def training_content(data):
    return html.Div([
        training_data_table(data),
        training_uploader(),
        html.Button('train model', id='train-button')
    ])

    
def setting_inputs():
    user_settings = get_user_settings()
    inputs = []
    for _, section in user_settings.items():
        for setting, value in section.items():
            inputs.append(html.Div(
                [
                    html.Label(setting),
                    dcc.Input(id=setting,value=value, placeholder=f'e.g. {value}')
                ]
            ))
    return inputs
    
def settings_content():
    # TODO: change input types to validate for filepath format
    return html.Div(
        id='settings-content',
        children=[
            *setting_inputs(),
            html.Button('save settings',id='save-settings'),
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


