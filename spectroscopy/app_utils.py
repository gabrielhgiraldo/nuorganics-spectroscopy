from configparser import ConfigParser
import logging
from pathlib import Path


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

def get_internal_settings():
    internal_config = ConfigParser()
    internal_config.read(INTERNAL_CONFIG_FILEPATH)
    return internal_config

def get_user_settings(setting=None):
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
    internal_settings = get_internal_settings()
    user_config_path = internal_settings.get('paths', 'user_configuration_path')
    user_config_path = Path(user_config_path)
    logger.info(f'new settings {new_settings}')
    logger.info(f'saving user settings at path {USER_CONFIG_PATH}')
    USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with USER_CONFIG_PATH.open('w') as f:
        user_config.write(f)

