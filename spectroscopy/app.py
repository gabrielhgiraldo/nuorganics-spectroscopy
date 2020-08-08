from pathlib import Path

import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

from spectroscopy.app_layout import (
    render_layout,
    settings_content,
    training_data_table,
    training_uploader,
    training_content,
)
from spectroscopy.app_utils import get_internal_settings, get_user_settings, save_user_settings
from spectroscopy.utils import load_training_data

# TODO: add ability to specify data location
# TODO: add ability to retrain model(s)
# TODO: add ability to configure included model parameters
# TODO: add ability to save retrained model(s)
# TODO: add ability to view and download prediction results

# load internal configurations
app = dash.Dash(__name__, suppress_callback_exceptions=True)
training_data = load_training_data()
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
        # TODO: load training data
        return training_content(training_data)
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

app.layout = render_layout

if __name__ == "__main__":
    app.server.run(debug=True)