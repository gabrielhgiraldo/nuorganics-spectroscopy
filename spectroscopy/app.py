import logging
import os
from threading import Timer
import webbrowser

import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, MATCH, Output, State
import matplotlib
matplotlib.use('agg')


from spectroscopy.app_layout import (
    inference_content,
    model_performance_section,
    render_layout,
    settings_content,
    training_content,
    model_data_table,
)
from spectroscopy.app_utils import (
    get_inference_data_path, get_model_dir,
    get_training_data_path,
    get_user_settings,
    inference_models,
    load_all_model_metrics,
    save_user_settings,
    upload_inference_data,
)
from spectroscopy.data import INFERENCE_RESULTS_FILENAME, SpectroscopyDataMonitor, EXTRACTED_DATA_FILENAME
from spectroscopy.model import train_models
## NEWEST TODO
# TODO: on start-up create folder structure
# TODO: browser freezing on load-up of data (paging?)

# TODO: implement file change detection system 
# TODO: keep track of loaded files
# TODO: add ability to configure included model parameters
# TODO: add ability to save retrained model(s)
# TODO: add ability to view and download prediction results
app = dash.Dash(__name__,
    title='Nuorganics Spectroscopy',
    suppress_callback_exceptions=True,
)
app.logger.setLevel(logging.INFO)
# create folder directories
get_training_data_path().mkdir(parents=True, exist_ok=True)

# initialize monitor for training data
training_data_monitor = SpectroscopyDataMonitor(
    watch_directory=get_training_data_path(),
    extracted_data_filename=EXTRACTED_DATA_FILENAME
)

inference_data_monitor = SpectroscopyDataMonitor(
    watch_directory=get_inference_data_path(),
    extracted_data_filename=INFERENCE_RESULTS_FILENAME
)


def get_triggered_id():
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    else:
        return ctx.triggered[0]['prop_id'].split('.')[0]

# tab callback
@app.callback(
    output=Output('tab-content', 'children'),
    inputs=[Input('tabs','value')])
def render_content(tab):
    if tab == 'settings-tab':
        return settings_content()
    elif tab == 'training-tab':
        return training_content()
    elif tab == 'inference-tab':
        return inference_content()
    else:
        return ['no content available for tab value']


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


# #upload data callback
# @app.callback(
#     output=Output('training-table-wrapper', 'children'),
#     inputs=[Input('upload-training', 'contents')],
#     state=[State('upload-training', 'filename')]
# )
# def update_training_data(contents, filenames):
#     if contents is not None and filenames is not None:
#         return [model_data_table(upload_training_data(contents, filenames), 'training')]
#     else:
#         try:
#             data, extracted_paths = load_training_data(skip_paths=extracted_paths)
#             return [model_data_table(data, 'training')]
#         except FileNotFoundError:
#             raise PreventUpdate
@app.callback(
    output=Output('training-table-wrapper', 'children'),
    inputs=[Input('training-data-sync-interval','n_intervals')]
)
def on_training_data_sync(num_training_syncs):
    app.logger.info('syncing data')
    # check if any of the files have changed and extract any that haven't
    if training_data_monitor.syncing:
        raise PreventUpdate
    if num_training_syncs:
        _, has_changed = training_data_monitor.sync_data()
        if not has_changed:
            raise PreventUpdate
    return [model_data_table(training_data_monitor.extracted_data, 'training')]


# train model callback
@app.callback(
    output=Output('model-metrics-wrapper', 'children'),
    inputs=[Input('train-models', 'n_clicks')],
    state=[State('training-target-selection', 'value')]
)
def on_train_models(n_clicks, training_targets):
    # TODO: get parameters for specific targets
    # TODO: get training data for specific targets
    # TODO: train only on a subset of the data based on the table state
    if n_clicks:
        model_dir = get_model_dir()
        train_models(training_targets, training_data_monitor.extracted_data, model_dir)
        # predict on test data and add it to the testing data section for exportation
        # generate testing data section
    model_metrics = load_all_model_metrics()
    return model_performance_section(model_metrics)

@app.callback(
    output=Output('inference-table-wrapper', 'children'),
    inputs=[Input('run-inference', 'n_clicks'),
            Input('upload-inference','contents')],
    state=[State('upload-inference', 'filename'),
           State('inference-target-selection','value')]
)
def on_inference(inference_clicks, contents, filenames, inference_targets):
    if contents and filenames:
        data, _ = upload_inference_data(contents, filenames)
    elif inference_clicks and inference_targets:
        data = inference_models(inference_targets, inference_data_monitor.extracted_data)
        inference_data_monitor.cache_data()
    else:
        try:
            data, has_change = inference_data_monitor.sync_data()
        except FileNotFoundError:
            raise PreventUpdate

    return model_data_table(data, 'inference')


# TODO: figure out how to select certain rows for export
# TODO: figure out how to select certains rows for inference?
# callback for syncing datatable data with storage
# @app.callback(
#     output=Output({"type":'total-samples', "index":MATCH},'children'),
#     inputs=[Input({"type":'data-table',"index":MATCH},'data')],
#     state=[State({"type":'data-table',"index":MATCH},'id')],
#     prevent_initial_call=True
# )
# def on_data_change(data, table_id):
#     if data is None:
#         raise PreventUpdate
#     tag = table_id['index']
#     # save data to 'tag' associated location
#     if tag == 'training':
#         # TODO: trigger confirmation?
#         training_data_path = get_training_data_path()
#         # pd.DataFrame(data).to_csv(training_data_path/EXTRACTED_DATA_FILENAME, index=False)
#     elif tag == 'inference':
#         inference_data_path = get_inference_data_path()
#         # pd.DataFrame(data).to_csv(inference_data_path/INFERENCE_RESULTS_FILENAME, index=False)
#     else:
#         raise PreventUpdate

#     return f'total samples: {len(data)}'




app.layout = render_layout


if __name__ == '__main__':
    port = 5000
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        Timer(1, lambda: webbrowser.open_new(f"http://localhost:{port}")).start()

    # Otherwise, continue as normal
    app.server.run(debug=True, port=port)