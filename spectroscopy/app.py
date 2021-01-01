import logging
import os
from threading import Timer
import webbrowser

import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, MATCH, Output, State
import matplotlib
matplotlib.use('agg')
import pandas as pd


from spectroscopy.app_layout import (
    model_performance_section,
    render_layout,
    model_data_table, transmittance_graph,
)
from spectroscopy.app_utils import (
    get_inference_data_path, 
    get_model_dir,
    get_training_data_path,
    get_user_settings,
    inference_models,
    save_user_settings,
    upload_inference_data,
)
from spectroscopy.data import INFERENCE_RESULTS_FILENAME, SpectroscopyDataMonitor, EXTRACTED_DATA_FILENAME, UnmatchedFilesException
from spectroscopy.modeling.dense_nn import define_model
from spectroscopy.modeling.utils import train_models, load_all_performance_artifacts
## NEWEST TODO
# TODO: add hyperparameter tuning to training pipeline
# TODO: allow ability to manage test set/ train-test split from UI or folders
# TODO: make script to correct file namings
# TODO: browser freezing on load-up of data (paging?)
# TODO: display model parameters
# TODO: give ability to choose which columns to include in extraction from lab report, etc.
# TODO: add ability to configure included model parameters


app = dash.Dash(__name__,
    title='Nuorganics Spectroscopy',
    suppress_callback_exceptions=True,
)
app.logger.setLevel(logging.INFO)

# initialize monitor for training data
training_data_monitor = SpectroscopyDataMonitor(
    watch_directory=get_training_data_path(),
    extracted_data_filename=EXTRACTED_DATA_FILENAME
)

# initialize monitor for inference data
inference_data_monitor = SpectroscopyDataMonitor(
    watch_directory=get_inference_data_path(),
    extracted_data_filename=INFERENCE_RESULTS_FILENAME
)
monitors = {
    'training':training_data_monitor,
    'inference': inference_data_monitor
}

def get_triggered_id():
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    else:
        return ctx.triggered[0]['prop_id'].split('.')[0]


# save settings callback
def generate_settings_callback_states():
    settings = get_user_settings()
    states = []
    for section in settings.values():
        for setting in section:
            state = State(setting, 'value')
            states.append(state)
    return states

    
@app.callback(
    output=Output('settings-feedback','children'),
    inputs=[Input('save-settings', 'n_clicks')],
    state=generate_settings_callback_states()
)
def on_save_settings(n_clicks, *args):
    input_id = get_triggered_id()
    if input_id == 'save-settings':
        new_setting_values = args
        try:
            save_user_settings(new_setting_values)
        except ValueError as e:
            return [f'{e}']
        else:
            return [f'settings saved']
    else:
        raise PreventUpdate       


# @app.callback(
#     output=Output('train-table-wrapper', 'children'),
#     inputs=[Input('train-data-sync-interval','n_intervals')]
# )
# def on_training_data_sync(num_training_syncs):
#     app.logger.info('syncing data')
#     # check if any of the files have changed and extract any that haven't
#     if training_data_monitor.syncing:
#         app.logger.warning(f'data still syncing, skipping sync')
#         return None
#     if num_training_syncs:
#         _, has_changed = training_data_monitor.sync_data()
#         if not has_changed:
#             raise PreventUpdate
#     return [model_data_table(training_data_monitor.extracted_data, 'train')]


# train model callback
@app.callback(
    output=[
        Output('model-metrics-wrapper', 'children'),
        Output('train-table-wrapper','children')
    ],
    inputs=[Input('train-models', 'n_clicks')],
    state=[State('train-target-selection', 'value')]
)
def on_train_models(n_clicks, training_targets):
    ctx = dash.callback_context
    changed_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if changed_id == 'train-models':
        model_dir = get_model_dir()
        artifacts, models = train_models(
            model_builder=define_model,
            targets=training_targets,
            data=training_data_monitor.extracted_data,
            model_dir=model_dir
        )
        # add predicted values for test samples
        for target, data_dict in artifacts['data'].items():
            # get training and testing data
            # mask = training_data_monitor.extracted_data.index.isin(data_dict['test_samples'].index)
            # training_data_monitor.extracted_data[mask]['train_test'] = 'test'
            # training_data_monitor.extracted_data[~mask]['train_test'] = 'train'
            # add predicted values to the data
            mask = training_data_monitor.extracted_data.index.isin(data_dict['y_pred'].index)
            training_data_monitor.extracted_data.loc[mask, f'predicted_{target}'] = data_dict['y_pred']
        # trigger ordering
        training_data_monitor.set_extracted_data(training_data_monitor.extracted_data)
    else:
        artifacts = load_all_performance_artifacts(model_dir=get_model_dir())
    return model_performance_section(artifacts), model_data_table(training_data_monitor.extracted_data, 'train')


@app.callback(
    output=Output('inference-table-wrapper', 'children'),
    inputs=[Input('run-inference', 'n_clicks'),
            Input('upload-inference','contents')],
    state=[State('upload-inference', 'filename'),
           State('inference-target-selection','value')]
)
def on_inference(inference_clicks, contents, filenames, inference_targets):
    ctx = dash.callback_context
    changed_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if changed_id == 'run-inference' and inference_clicks and inference_targets:
        data = inference_models(inference_targets, inference_data_monitor.extracted_data)
        inference_data_monitor.set_extracted_data(data)
    elif contents and filenames:
        upload_inference_data(contents, filenames)
        inference_data_monitor.sync_data()
    else:
        try:
           inference_data_monitor.sync_data()
        except FileNotFoundError:
            raise PreventUpdate

    return model_data_table(inference_data_monitor.extracted_data, 'inference')

@app.callback(
    output=Output({'type':'scan-viewer', 'index':MATCH}, 'children'),
    inputs=[Input({'type':'view-scans', 'index':MATCH}, 'n_clicks')],
    state=(
        State({'type':'data-table','index':MATCH}, 'derived_virtual_data'),
        State({'type':'data-table','index':MATCH}, 'derived_virtual_selected_rows'),
    )
)
def on_view_scans(scan_clicks, data, selected_row_indices):
    if selected_row_indices is None:
        raise PreventUpdate
    data = pd.DataFrame(data).iloc[selected_row_indices]
    return transmittance_graph(data)


# TODO: add on_filter for datatable to update total samples


def _on_refresh():
    # create folder directories
    get_training_data_path().mkdir(parents=True, exist_ok=True)
    get_inference_data_path().mkdir(parents=True, exist_ok=True)
    get_model_dir().mkdir(parents=True, exist_ok=True)
    return render_layout(
        training_monitor=training_data_monitor,
        inference_monitor=inference_data_monitor
    )

app.layout = _on_refresh


if __name__ == '__main__':
    port = 5000
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        Timer(1, lambda: webbrowser.open_new(f"http://localhost:{port}")).start()

    # Otherwise, continue as normal
    app.server.run(debug=True, port=port)