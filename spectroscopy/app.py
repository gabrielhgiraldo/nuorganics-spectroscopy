from pathlib import Path

import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
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
from spectroscopy.app_utils import get_user_settings, save_user_settings

# TODO: add ability to specify data location
# TODO: add ability to retrain model(s)
# TODO: add ability to configure included model parameters
# TODO: add ability to save retrained model(s)
# TODO: add ability to view and download prediction results
# TODO: determine configuration flow
# TODO: if no configuration file is found, prompt user to input some configurations

# load internal configurations
app = dash.Dash(__name__, suppress_callback_exceptions=True)

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
        return training_content(pd.DataFrame([range(5)]*5,columns=range(5)))
    else:
        return ['no content available for tab value']

# save settings callback
def generate_settings_callback_inputs():
    settings = get_user_settings()
    inputs = [Input('settings-save-button', 'num_clicks')]
    for section in settings.values():
        for setting in section:
            inpt = Input(setting, 'value')
            inputs.append(inpt)
    return inputs

    
@app.callback(
    output=Output('settings-feedback','children'),
    inputs=generate_settings_callback_inputs()
)
def on_save(num_clicks, *args):
    print('save_settings triggered', num_clicks, *args)
    input_id = get_triggered_id()
    if input_id == 'settings-save-button':
        messages = []
        # TODO: loop through settings in user_config and check
        # if settings_filepath is None or training_directory is None:
        #     if settings_filepath is None:
        #         messages.append(html.P('settings filepath required'))
        #     if training_directory is None:
        #         messages.append(html.P('training directory required'))
        #     return messages
        # else:
        #     # TODO: check if paths are valid
        #     new_settings = 
        #     validate_settings()
           
        #     save_user_settings(new_settings)
        #     messages.append('settings saved')
            # return messages
        return messages
        # TODO: save settings to configuration file
    else:
        raise PreventUpdate       

app.layout = render_layout

if __name__ == "__main__":
    app.server.run(debug=True)