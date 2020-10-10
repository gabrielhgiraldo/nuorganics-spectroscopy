import base64
from configparser import ConfigParser
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

from spectroscopy.model import load_model, transform_data
from spectroscopy.data import (
    AVAILABLE_TARGETS,
    INFERENCE_RESULTS_FILENAME, UnmatchedFilesException,
    extract_data,
)
USER_CONFIG_PATH = Path('config.ini')
DEFAULT_USER_CONFIGS = {
    'paths':{
        'project-path':str(Path().home()/'spectroscopy'),
        'data-path':'%(project-path)s/data',
        # 'training-data-path':'%(data-path)s/training',
        # 'testing-data-path':'%(data-path)s/testing',
        'models-path':'%(project-path)s/models',
        'results-data-path':'%(project-path)s/results',
    }
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_user_settings():
    user_config = ConfigParser()
    # set default settings in case no config file is found
    user_config.read_dict(DEFAULT_USER_CONFIGS)
    if USER_CONFIG_PATH.exists():
        user_config.read(USER_CONFIG_PATH)
    else:
        logger.warn(f'no configuration file found at {USER_CONFIG_PATH}')
    return user_config


def save_user_settings(new_settings_values):        
    user_config = get_user_settings()
    # TODO: deal with sections
    if isinstance(new_settings_values, (list, tuple)):
        new_settings_values = list(new_settings_values)
        new_settings = {}
        for section_name, section in user_config.items():
            new_settings[section_name] = {}
            for setting in section:
                new_settings[section_name][setting] = new_settings_values.pop(0)
    else:
        new_settings = new_settings_values
    # validate settings
    for section_name, section in new_settings.items():
        for setting, value in section.items():
            if not value:
                raise ValueError(f'{setting} is required')
    
    user_config.update(new_settings)
    logger.info(f'new settings {new_settings}')
    logger.info(f'saving user settings at path {USER_CONFIG_PATH}')
    USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with USER_CONFIG_PATH.open('w') as f:
        user_config.write(f)

# TODO: generalize these and include in custom upload_data_section component
def get_project_path():
    return Path(get_user_settings()['paths']['project-path'])


def get_all_data_path():
    return Path(get_user_settings()['paths']['data-path'])


def get_training_data_path():
    return get_all_data_path()


def get_inference_data_path():
    return Path(get_user_settings()['paths']['results-data-path'])


def get_model_dir():
    return Path(get_user_settings()['paths']['models-path'])


def upload_data(path, contents, filenames):
    """save byte data to folder"""
    path.mkdir(exist_ok=True, parents=True)
    for content, filename in zip(contents, filenames):
        content_type, _, content_string = content.partition(',')
        decoded = base64.b64decode(content_string)
        with open(path/filename, 'wb') as f:
            f.write(decoded)


def upload_training_data(contents, filenames, skip_paths=None):
    training_data_path = get_training_data_path()
    return upload_data(training_data_path, contents, filenames, skip_paths=skip_paths)


def upload_inference_data(contents, filenames):
    inference_data_path = get_inference_data_path()     
    return upload_data(
        path=inference_data_path,
        contents=contents,
        filenames=filenames,
    )  


def load_models(tags):
    if tags is None:
        tags = AVAILABLE_TARGETS
    models = {}
    model_dir = get_model_dir()
    for tag in tags:
        try:
            models[tag] = load_model(tag, model_dir)
        except FileNotFoundError:
            logger.warn(f'no model {tag} found in dir {model_dir}')
    return models

# TODO: speed up inference of models with concurrency
def inference_models(model_tags, data):
    # if data is None:
    #     data = load_inference_data()
    models = load_models(model_tags)
    X = transform_data(data)
    for model_tag, model in models.items():
        logger.info(f'running inference with model {model_tag}')
        data[f'predicted_{model_tag}'] = model.predict(X)
        data[f'predicted_on'] = pd.to_datetime(datetime.now())
    return data

