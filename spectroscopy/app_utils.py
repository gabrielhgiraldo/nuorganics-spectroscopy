from configparser import ConfigParser
import logging
from pathlib import Path

INTERNAL_CONFIG_FILEPATH = Path(__file__).parent / 'config.ini'
DEFAULT_USER_CONFIGS = {
    'paths':{
        'project-path':'~/spectroscopy',
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


def get_user_settings():
    user_config = ConfigParser()
    # set default settings in case no config file is found
    user_config.read_dict(DEFAULT_USER_CONFIGS)
    user_config_path = get_internal_settings().get('paths', 'user_configuration_path')
    user_config_path = Path(user_config_path)
    if user_config_path.exists():
        user_config.read(user_config_path)
    else:
        logger.warn(f'no configuration file found at {user_config_path}')
    return user_config


def save_user_settings(new_settings):
    user_config = get_user_settings()
    user_config.update(new_settings)
    internal_settings = get_internal_settings()
    user_config_path = internal_settings.get('paths', 'user_configuration_path')
    user_config_path = Path(user_config_path)
    user_config.write(user_config_path)

