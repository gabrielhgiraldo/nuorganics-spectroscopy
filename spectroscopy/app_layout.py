import logging

import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable


from spectroscopy.app_utils import get_user_settings
from spectroscopy.utils import (
    AVAILABLE_TARGETS,
)


TARGET_OPTIONS = [{'label':target, 'value':target} for target in AVAILABLE_TARGETS]
logger = logging.getLogger(__name__)

# TODO: fix certain columns on the left
# TODO: order columns
# TODO: include wavelength columns in export
# TODO: split layout amongst tab sections?

## GENERAL
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
        html.Div(id='tab-content', className='container')
    ])

## SETTINGS
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
        ],
        className='container'
    )
## TRAININING
def model_data_table(data, tag):
    return html.Div([
        DataTable(
            id=f'{tag}-data-table',
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



def training_target_selector():
    # checkboxes for which models to do inference with
    options = [{**target, 'disabled':False} for target in TARGET_OPTIONS]
    return html.Div([
        dcc.Checklist(
            id='training-target-selection',
            options=options,
            labelStyle={'display':'inline-block'},
            style={'display':'inline-block'}
        ),
        html.P([html.Small('select models to train')]),
    ])

# TODO: create custom component for this
# TODO: make these collapsible 
def model_data_section(tag):
    return html.Div(
        children=[
            html.H4(f'{tag} Data'),
            html.Small(f'data to be used for {tag} models'),
            dcc.Upload(
                id=f'upload-{tag}',
                multiple=True,
                children=[
                    html.Button(
                        'Upload Files',
                    ),
                ],
                style={'float':'left'}
            ),
            dcc.Loading(id=f'{tag}-table-wrapper'),
        ],
    )


def trained_models_section():
    return html.Div(
        children=[
            html.H4('Models', className='u-cf'),
            training_target_selector(),
            html.Button(
                'train model(s)',
                id='train-models',
                n_clicks=0,
            ),
            dcc.Loading(id='model-metrics-wrapper')
        ],
    )


# a card for each model with the model metrics
def model_metrics_section(model_metrics):
    # load model metrics
    # create card for every model
    metrics_cards = []
    for model, metrics in model_metrics.items():
        card = html.Div(
            children=[str(metrics)]
        )
        metrics_cards.append(card)

    return html.Div(
        children=metrics_cards
    )


# TODO: adjust column ordering in training datatable
def training_content():
    return html.Div(
        children=[
            dcc.Loading(id='training-feedback'),
            model_data_section('training'),
            # model_data_section('testing'),
            trained_models_section(),
        ],
    )

    


## INFERENCE

def inference_target_selector():
    # checkboxes for which models to do inference with
    # TODO: enable targets based on  available trained models
    # loop through targets and see which ones have models saved in folders
    options = [{**target, 'disabled':True} for target in TARGET_OPTIONS]
    return dcc.Checklist(
        id='inference-target-selection',
        options=options,
        labelStyle={'display':'inline-block'},
        style={'display':'inline-block'}
    )

# TODO: add table for uploading samples for inference
def inference_content():
    return html.Div([
        # select which targets you want to include in inference (which models)
        inference_target_selector(),
        # button for running inference
        html.Button('run inference', id='run-inference', n_clicks=0)
    ])

