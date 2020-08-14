import os
from pathlib import Path

from threading import Timer
import webbrowser


import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import matplotlib
matplotlib.use('agg')

from spectroscopy.app_layout import (
    inference_content,
    render_layout,
    settings_content,
    training_content,
    training_data_table,
)
from spectroscopy.app_utils import (
    get_user_settings,
    load_training_data, 
    save_user_settings,
    upload_training_data,
)
from spectroscopy.model import train_models
from spectroscopy.utils import get_wavelength_columns



# TODO: add ability to specify data location
# TODO: add ability to retrain model(s)
# TODO: add ability to configure included model parameters
# TODO: add ability to save retrained model(s)
# TODO: add ability to view and download prediction results

# load internal configurations
app = dash.Dash(__name__,
    title='Nuorganics Spectroscopy',
    suppress_callback_exceptions=True,
    # external_stylesheets=[dbc.themes.MINTY],
    # external_stylesheets=[dbc.themes.MATERIA]
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
)

# TODO: add ability to change configurations
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
def on_save(n_clicks, *args):
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

# upload data callback
@app.callback(
    output=Output('training-table-wrapper', 'children'),
    inputs=[Input('upload-training', 'contents')],
    state=[State('upload-training', 'filename'),
           State('upload-training', 'last_modified')])
def update_training_data(contents, filenames, last_modifieds):
    if contents is not None and filenames is not None:
        return [training_data_table(upload_training_data(contents, filenames))]
    else:
        try:
            data = load_training_data()
            data = data.drop(get_wavelength_columns(data), axis=1)
            return [training_data_table(data)]
        except FileNotFoundError:
            raise PreventUpdate


# TODO: add ability to choose model type
# TODO: add loading spinner while models are training
# train model callback
@app.callback(
    output=Output('training-feedback', 'children'),
    inputs=[Input('train-model', 'n_clicks')],
    state=[State('training-target-selection', 'value')]
)
def on_train_model(n_clicks, training_targets):
    # TODO: get parameters for specific targets
    # TODO: get training data for specific targets
    if not n_clicks:
        raise PreventUpdate
    model_dir = Path(get_user_settings()['paths']['project-path'])
    train_models(training_targets, model_dir)
    # TODO: display model training results
    return ['models trained']

    

app.layout = render_layout


if __name__ == '__main__':
    port = 5000
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        Timer(1, lambda: webbrowser.open_new(f"http://localhost:{port}")).start()

    # Otherwise, continue as normal
    app.server.run(debug=True, port=port)