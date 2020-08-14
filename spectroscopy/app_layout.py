import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable


from spectroscopy.app_utils import get_user_settings
from spectroscopy.utils import AMMONIA_N, PERCENT_MOISTURE, PHOSPHORUS, POTASSIUM, SULFUR



AVAILABLE_TARGETS = [
    {'label':'AMMONIA-N','value':AMMONIA_N},
    {'label':'PERCENT MOISTURE', 'value':PERCENT_MOISTURE},
    {'label':'PHOSPHOROUS', 'value':PHOSPHORUS},
    {'label':'POTASSIUM', 'value':POTASSIUM},
    {'label':'SULFUR', 'value':SULFUR}
]

# TODO: fix certain columns on the left
# TODO: order columns
def training_data_table(data):
    return html.Div([
        # html.H3('Extracted Data'),
        DataTable(
            id='training-data-table',
            columns=[{"name": column, "id": column} for column in data.columns],
            data=data.to_dict('records'),
            fixed_rows={'headers': True},
            sort_action='native',
            export_format='xlsx',
            style_table={
                # 'width':'100%',
                'height': '800px',
                'overflowY': 'auto',
                'overflowX':'auto'
            },
            style_cell={
                # all three widths are needed
                'minWidth': '180px',
                'width': '180px',
                'maxWidth': '180px',
                #text overflow settings
                'overflow':'hidden',
                'textOverflow':'ellipsis',
                'textAlign':'left'
            },
            tooltip_data=[
                {
                    column: {'value': str(value), 'type': 'markdown'}
                    for column, value in row.items()
                } for row in data.to_dict('rows')
            ],
            tooltip_duration=None,
            tooltip_delay=800
        ),
        html.P(f'total: {len(data.index)} samples'),
        ],
        style={'textAlign':'center'}
    )


# TODO: adjust column ordering in training datatable
def training_content():
    return html.Div([
        dcc.Loading(id='training-feedback'),
        html.Div([
            training_target_selector(),
            dcc.Upload(
                id='upload-training',
                multiple=True,
                children=[
                    html.Button(
                        'Upload Files',
                    ),
                ],
            ),
            html.Button(
                'train model',
                id='train-model',
                n_clicks=0,
            ),
        ],
        style={'textAlign':'center', 'padding':'10px'}),
        dcc.Loading(id='training-table-wrapper'),
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
            html.Button('save settings', id='save-settings', n_clicks=0),
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


def inference_target_selector():
    # checkboxes for which models to do inference with
    options = [{**target, 'disabled':True} for target in AVAILABLE_TARGETS]
    return dcc.Checklist(
        id='inference-target-selection',
        options=options,
        labelStyle={'display':'inline-block'},
        style={'display':'inline-block'}
    )


def training_target_selector():
    # checkboxes for which models to do inference with
    options = [{**target, 'disabled':False} for target in AVAILABLE_TARGETS]
    return dcc.Checklist(
        id='training-target-selection',
        options=options,
        labelStyle={'display':'inline-block'},
        style={'display':'inline-block'}
    )


def inference_content():
    return html.Div([
        # select which targets you want to include in inference (which models)
        inference_target_selector('inference-target-selection'),
        # button for running inference
        html.Button('run inference', id='run-inference', n_clicks=0)
    ])

