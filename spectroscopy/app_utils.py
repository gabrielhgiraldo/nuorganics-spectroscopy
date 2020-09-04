import base64
from configparser import ConfigParser
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

from spectroscopy.model import load_model, load_model_metrics, transform_data
from spectroscopy.utils import (
    AVAILABLE_TARGETS,
    INFERENCE_RESULTS_FILENAME,
    TRAINING_DATA_FILENAME,
    extract_data,
    load_extracted_data
)

INTERNAL_CONFIG_FILEPATH = Path(__file__).parent / 'config.ini'
USER_CONFIG_PATH = Path('config.ini')
DEFAULT_USER_CONFIGS = {
    'paths':{
        'project-path':str(Path().home()/'spectroscopy'),
        'data-path':'%(project-path)s/data',
        # 'training-data-path':'%(data-path)s/training',
        # 'testing-data-path':'%(data-path)s/testing',
        'results-data-path':'%(project-path)s/results',
    }
}

logger = logging.getLogger(__name__)


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
def get_all_data_path():
    return Path(get_user_settings()['paths']['data-path'])

def get_training_data_path():
    return get_all_data_path()


def get_inference_data_path():
    return Path(get_user_settings()['paths']['results-data-path'])


def load_data(data_path, filename=None):
    try:
        logger.info('loading extracted data')
        return load_extracted_data(data_path, filename)
    except FileNotFoundError:
        message = (
            f'no previously extracted data found at {data_path}'
            '\n extracting data from raw files'
        )
        logger.warning(message)
        try:
            return extract_data(data_path, filename)
        except FileNotFoundError as e:
            logger.warning(e)
            raise


def load_training_data():
    training_data_path = get_training_data_path()
    return load_data(training_data_path)


def load_inference_data():
    inference_data_path = get_inference_data_path()
    return load_data(inference_data_path, INFERENCE_RESULTS_FILENAME)


def upload_data(path, contents, filenames):
    path.mkdir(exist_ok=True, parents=True)
    for content, filename in zip(contents, filenames):
        content_type, _, content_string = content.partition(',')
        decoded = base64.b64decode(content_string)
        with open(path/filename, 'wb') as f:
            f.write(decoded)
    data = extract_data(path, TRAINING_DATA_FILENAME)
    return data


def upload_training_data(contents, filenames):
    training_data_path = get_training_data_path()
    return upload_data(training_data_path, contents, filenames)


def upload_inference_data(contents, filenames):
    inference_data_path = get_inference_data_path()     
    return upload_data(inference_data_path, contents, filenames)  


def get_model_dir():
    return Path(get_user_settings()['paths']['project-path'])


def load_all_model_metrics():
    model_dir = get_model_dir()
    metrics = {}
    for target in AVAILABLE_TARGETS:
        try:
            model_metrics = load_model_metrics(target, model_dir)
        except FileNotFoundError as e:
            logger.warning(e)
        else:
            metrics[target] = model_metrics
    return metrics


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


def inference_models(model_tags, data=None, results_path=None):
    if results_path is None:
        results_path = get_inference_data_path()
    if data is None:
        data = load_inference_data()

    models = load_models(model_tags)
    X = transform_data(data)
    for model_tag, model in models.items():
        logger.info(f'running inference with model {model_tag}')
        data[f'predicted_{model_tag}'] = model.predict(X)
        data[f'predicted_on'] = pd.to_datetime(datetime.now())
    data.to_csv(results_path/INFERENCE_RESULTS_FILENAME, index=False)
    return data

