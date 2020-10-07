import logging

import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable
from dash_table.Format import Format, Scheme
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime, is_numeric_dtype




from spectroscopy.app_utils import get_user_settings
from spectroscopy.data import AVAILABLE_TARGETS, SCAN_FILE_DATETIME_FORMAT
from spectroscopy.utils import get_wavelength_columns

ENABLE_UI_UPLOAD = False
METRIC_PRECISION = 2
TABLE_NUMERIC_PRECISION = 2
TRAINING_SYNC_INTERVAL = 20000
TARGET_OPTIONS = [{'label':target, 'value':target} for target in AVAILABLE_TARGETS]
logger = logging.getLogger(__name__)

# TODO: fix certain columns on the left
# TODO: order columns
# TODO: include wavelength columns in export
# TODO: split layout amongst tab sections?
# TODO: add loading messages to loaders

## GENERAL
def render_layout():
    # create data folder and parents
    return html.Div([
        html.H3('Nuorganics Spectroscopy Modeling'),
        dcc.Tabs(
            id='tabs',
            value='inference-tab',
            persistence=True,
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
    columns = []
    for column in data.columns:
        column_config = {
            'name':column,
            'id':column,
        }
        # add formatting for numeric columns
        if is_numeric_dtype(data[column]):
            # data[column] = data[column].round(TABLE_NUMERIC_PRECISION)
            column_config.update({
                'type':'numeric',
                'format':Format(
                    precision=TABLE_NUMERIC_PRECISION,
                    scheme=Scheme.fixed
                )
            })
        elif is_datetime(data[column]) or column.endswith('date'):
            data[column] = pd.to_datetime(
                arg=data[column],
                format=SCAN_FILE_DATETIME_FORMAT,
                errors='coerce'
            ).dt.strftime(SCAN_FILE_DATETIME_FORMAT)
        columns.append(column_config)
    return html.Div([
        DataTable(
            id={
                "type":'data-table',
                "index":tag
            },
            # columns=[{"name": column, "id": column} for column in data.columns],
            columns=columns,
            data=data.to_dict('records'),
            fixed_rows={'headers': True},
            # fixed_columns={'headers':True},
            sort_action='native',
            export_format='xlsx',
            persistence=True,
            # row_deletable=True,
            # row_selectable=True,
            # TODO: add option to include wavelength columns in export
            hidden_columns=get_wavelength_columns(data),
            css=[{"selector": ".show-hide", "rule": "display: none"}],
            # export_columns='all',
            style_table={
                # 'width':'100%',
                'max-height': '800px',
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
        html.P(f'total: {len(data.index)} samples', id={"type":'total-samples', "index":tag}),
        ],
        style={'textAlign':'center'}
    )


def target_selector(tag):
    options = [{**target, 'disabled':False} for target in TARGET_OPTIONS]
    return html.Div([
        dcc.Checklist(
            id=f'{tag}-target-selection',
            options=options,
            labelStyle={'display':'inline-block'},
            style={'display':'inline-block'},
            value=AVAILABLE_TARGETS
        ),
        html.P([html.Small(f'select models to {tag}')]),
    ])


# TODO: create custom component for this
# TODO: make these collapsible 
# TODO: accept only certaint tpyes of files as variable
def model_data_section(tag, sync_interval=60000, enable_upload=True):
    return html.Div(
        children=[
            html.H4(f'{tag} Data'),
            html.Small(f'data to be used for {tag} models'),
            dcc.Interval(id=f'{tag}-data-sync-interval', interval=sync_interval),
            dcc.Upload(
                id=f'upload-{tag}',
                multiple=True,
                children=[
                    html.Button(
                        'Upload Files',
                    ),
                ],
                style={
                    'float':'left',
                    'z-index':1,
                    'position':'relative',
                    'visibility':'visible' if enable_upload else 'hidden'
                }
            ),
            dcc.Loading(id=f'{tag}-table-wrapper'),
        ],
        className="row"
    )


def trained_models_section():
    return html.Div(
        children=[
            html.H4('Models'),
            target_selector('training'),
            html.Button(
                'train model(s)',
                id='train-models',
                n_clicks=0,
            ),
            dcc.Loading(id='model-metrics-wrapper')
        ],
    )

def predicted_vs_actual_graph():
    pass


def metric_card(metric_name, metric_value, n_columns=1):
    word_to_number = ["zero","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve"]
    return html.Div(
        children=[
            html.Small(f"{metric_name}\n"),
            html.Small(html.B(round(metric_value, METRIC_PRECISION)))
        ],
        className=f"{word_to_number[int(12/n_columns)]} columns"
    )

def metrics_section(section_name, metrics):
    cards = []
    for metric, metric_value in metrics.items():
        card = metric_card(metric, metric_value, len(metrics))
        cards.append(card)
    
    return html.Div([
        html.B(html.Small(section_name)),
        html.Div(cards, className="row")
    ])

# TODO: add model performance graphs here?
# TODO: split metrics into sections (train vs test)
def model_card(model_tag, metrics):
    metrics_sections = []
    for section, section_metrics in metrics.items():
        metrics_sections.append(metrics_section(section, section_metrics))

    return html.Div(
        children=[
            html.B(model_tag),
            html.Div(
                children=metrics_sections,
                className="row model-card"
            )
        ]
    )


# TODO: include residual graphs, fit graphs, other graphs,
# TODO: include maximum value, minimum value for each metric, stdev, etc.
def model_performance_section(artifacts):
    model_metrics = artifacts['metrics']
    metrics_cards = []
    for model, metrics in model_metrics.items():
        card = model_card(model, metrics)
        metrics_cards.append(card)
    model_graphs = artifacts['graphs']
    # TODO: include model graphs
    # for model, graphs in model_graphs.items():
    #     graph = mode
    return html.Div(
        children=[html.H5('Performance'), *metrics_cards]
    )


# TODO: adjust column ordering in training datatable
# TODO: add testing data section and results
def training_content():
    return html.Div(
        children=[
            dcc.Loading(id='training-feedback'),
            model_data_section('training',
                enable_upload=False,
                sync_interval=TRAINING_SYNC_INTERVAL
            ),
            trained_models_section(),
        ],
    )


## INFERENCE
def inference_content():
    return html.Div([
        model_data_section('inference'),
        target_selector('inference'),
        html.Button('run inference', id='run-inference', n_clicks=0)
    ])

