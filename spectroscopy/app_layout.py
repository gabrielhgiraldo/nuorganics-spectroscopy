import logging

import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable
from dash_table.Format import Format, Scheme
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime, is_numeric_dtype
import plotly.express as px
import plotly.graph_objects as go




from spectroscopy.app_utils import get_user_settings, img_path_to_base64
from spectroscopy.data import AVAILABLE_TARGETS, INFERENCE_RESULTS_FILENAME, SCAN_IDENTIFIER_COLUMNS, SCAN_FILE_DATETIME_FORMAT
from spectroscopy.utils import get_wavelength_columns

TRAINING_SYNC_INTERVAL = 60000
INFERENCE_SYNC_INTERVAL = 60000
METRIC_PRECISION = 2
TABLE_NUMERIC_PRECISION = 2
TARGET_OPTIONS = [{'label':target, 'value':target} for target in AVAILABLE_TARGETS]
logger = logging.getLogger(__name__)

# TODO: fix certain columns on the left
# TODO: add loading messages to loaders

## GENERAL
def render_layout(training_monitor, inference_monitor):
    # create data folder and parents
    return html.Div([
        html.H3('Nuorganics Spectroscopy Modeling'),
        dcc.Tabs(
            id='tabs',
            value='inference-tab',
            persistence=True,
            children=[
                dcc.Tab(label='Settings', value='settings-tab',
                    children=settings_content()
                ),
                dcc.Tab(label='Training', value='training-tab',
                    children=training_content(training_monitor)    
                ),
                dcc.Tab(label='Inference', value='inference-tab',
                    children=inference_content(inference_monitor)
                ),
                # dcc.Tab(label='Analysis', value='analysis-tab')
            ]),
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
        elif is_datetime(data[column]) or str(column).endswith('date'):
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
            # editable=True,
            filter_action="native",
            sort_mode="multi",
            export_format='xlsx',
            persistence=True,
            # row_deletable=True,
            row_selectable='multi',
            selected_rows=[],
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
        html.P([html.Small(f'select models for {tag}')]),
    ])


# TODO: create custom component for this
# TODO: make these collapsible 
# TODO: accept only certaint tpyes of files as variable
def model_data_section(tag, monitor, sync_interval=None, enable_upload=True):
    monitor.sync_data()
    children = [
            html.H4(f'{tag} Data'.title()),
            html.Small(f'data used to {tag} models'),
            dcc.Loading(id={'type':'scan-viewer','index':tag}),
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
                    'zIndex':1,
                    'position':'relative',
                    'display':'block' if enable_upload else 'none'
                }
            ),
            html.Button(
                'view scans',
                id={'type':'view-scans', 'index':tag},
                style={
                    'float':'left',
                    'zIndex':1,
                    'position':'relative',
                }            
            ),
            dcc.Loading(id=f'{tag}-table-wrapper',
                children=model_data_table(monitor.extracted_data, tag)
            ),
    ]
    if sync_interval is not None:
        children.append(
            dcc.Interval(
                id=f'{tag}-data-sync-interval',
                interval=sync_interval
            ),
        )
    return html.Div(
        children=children,
        className="row"
    )


def trained_models_section():
    return html.Div(
        children=[
            html.H4('Models'),
            target_selector('train'),
            html.Button(
                'train model(s)',
                id='train-models',
                n_clicks=0,
            ),
            # TODO: fix placement of this so it doesn't move off screen
            dcc.Loading(id='model-metrics-wrapper')
        ],
    )
# TODO: on hover, give sample information
def pred_v_actual_graph(samples, target, y_pred, y_true):
    max_value = y_true.max()
    max_value += max_value/10
    
    fig = px.scatter(
        data_frame=samples,
        x=y_true,
        y=y_pred,
        title=f'{target} predicted vs actual',
        opacity=0.7,
        range_x=[0, max_value],
        range_y=[0, max_value],
        width=800+200,
        height=800,
        labels={
            'x':f'True {target}',
            'y':f'Predicted {target}',
        },
        # color='sample_name',
        # color='process_method',
        color_discrete_sequence=px.colors.qualitative.Vivid,
        hover_data=SCAN_IDENTIFIER_COLUMNS
    )
    # add ideal fit line
    x = np.round(np.linspace(0, max_value, len(y_true)),2)
    fig.add_trace(px.line(
        x=x,
        y=x,
    ).data[0])
    return dcc.Graph(figure=fig)



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
def model_performance_section(artifacts, interactive_graph=True):
    # metrics
    children = [html.H5('Performance')]
    model_metrics = artifacts['metrics']
    metrics_cards = []
    for model, metrics in model_metrics.items():
        card = model_card(model, metrics)
        metrics_cards.append(card)
    children.extend(metrics_cards)
    # graphs
    if interactive_graph:
        for target, data_dict in artifacts['data'].items():
            y_pred = data_dict['y_pred']
            y_true = data_dict['y_test']
            samples = data_dict['test_samples']
            graph = pred_v_actual_graph(samples, target, y_pred, y_true)
            children.append(graph)

    else:
        # graphs as imgs
        model_graph_paths = artifacts['graphs']
        graph_imgs = []
        for model, graph_paths in model_graph_paths.items():
            model_graphs = [img_path_to_base64(path) for path in graph_paths]
            model_graph_imgs = []
            for graph in model_graphs:
                graph_img = html.Img(
                    src='data:image/png;base64,{}'.format(graph),
                )
                model_graph_imgs.append(graph_img)
            graph_imgs.extend(model_graph_imgs)
        children.extend(graph_imgs)
    return html.Div(
        children=children
    )


# TODO: add testing data section and results
def training_content(monitor, sync_interval=None):
    return html.Div(
        children=[
            model_data_section('train',
                enable_upload=False,
                sync_interval=sync_interval,
                monitor=monitor
            ),
            trained_models_section(),
        ],
        className='container'
    )


## INFERENCE
def inference_content(monitor, sync_interval=None):
    return html.Div(
        children=[
            model_data_section('inference',
                monitor=monitor,
                sync_interval=sync_interval,
                enable_upload=False
            ),
            target_selector('inference'),
            html.Button('run inference', id='run-inference', n_clicks=0)
        ],
        className='container'
    )


def transmittance_graph(data):
    wavelengths = get_wavelength_columns(data)
    wavelength_data = data[wavelengths]
    fig = go.Figure()
    for i in wavelength_data.index:
        x = [float(wavelength) for wavelength in wavelengths]
        y = wavelength_data.loc[i].to_list()
        fig.add_trace(
            go.Line(
                # data_frame=data.loc[i],
                x=x,
                y=y,
                name=str(data.loc[i]['index']),
                # name='index',
                
                # hover_data=['index']
                # hover_name='index'
                # hover_name=str(data.loc[i]['index']),
                # hover_data=str(data.loc[i]['index'])
            )
        )
    fig.update_layout(
        # title="Transmittance",
        xaxis_title="Wavelength",
        yaxis_title="Transmittance",
        legend_title="sample",
        showlegend=True,
        legend=dict(
            y=1,
            x=1
        ),
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        #     color="RebeccaPurple"
        # )
    )
    return dcc.Graph(figure=fig)
