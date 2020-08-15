import base64
from configparser import ConfigParser
import logging
from pathlib import Path

from spectroscopy.model import load_model_metrics
from spectroscopy.utils import (
    AVAILABLE_TARGETS,
    extract_data,
    get_wavelength_columns,
    load_extracted_training_data
)

INTERNAL_CONFIG_FILEPATH = Path(__file__).parent / 'config.ini'
USER_CONFIG_PATH = Path('config.ini')
DEFAULT_USER_CONFIGS = {
    'paths':{
        'project-path':str(Path().home()/'spectroscopy'),
        'data-path':'%(project-path)s/data',
        'training-data-path':'%(data-path)s/training',
        'testing-data-path':'%(data-path)s/testing',
        'output-path':'%(project-path)s/results',
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


def get_training_data_path():
    return Path(get_user_settings()['paths']['training-data-path'])

    
def load_training_data():
    training_data_path = get_training_data_path()
    try:
        logger.info('loading extracted data')
        return load_extracted_training_data(training_data_path)
    except FileNotFoundError:
        message = (
            f'no previously extracted data found at {training_data_path}'
            '\n extracting data from raw files'
        )
        logger.warning(message)
        try:
            return extract_data(training_data_path)
        except FileNotFoundError as e:
            logger.warning(e)
            raise


def upload_training_data(contents, filenames):
    training_data_path = get_training_data_path()
    training_data_path.mkdir(exist_ok=True, parents=True)
    for content, filename in zip(contents, filenames):
        content_type, _, content_string = content.partition(',')
        decoded = base64.b64decode(content_string)
        with open(training_data_path/filename, 'wb') as f:
            f.write(decoded)
    data = extract_data(training_data_path)
    data = data.drop(get_wavelength_columns(data), axis=1)
    return data
        

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
# def get_samples_dir():

# def load_models():
#     models = {}
#     model_dir = get_model_dir()
#     for target in AVAILABLE_TARGETS:
#         try:
#             models['target'] = load_model(target, model_dir)
#         except FileNotFoundError:
#             logger.warn(f'no model for target {target} found in dir {model_dir}')
