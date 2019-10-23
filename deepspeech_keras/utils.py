import os
import logging
from logging import Logger
from typing import Callable, Any
import dill


def save(data: Any, file_name: str):
    """ Save arbitrary data in the file. """
    with open(file_name, mode='wb') as file:
        dill.dump(data, file)


def get_root_dir() -> str:
    source_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(source_dir, '..'))


def chdir(to='ROOT'):
    """ Change the current work directory. Easily it can be switched to the project ROOT."""
    new_cwd = get_root_dir() if to == 'ROOT' else to
    os.chdir(new_cwd)


def create_logger(file_path='', level=20, name='deepspeech') -> Logger:
    """ Create the logger and handlers both console and file. """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formater = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formater)
    logger.addHandler(console)       # handle all messages from logger (not set handler level)
    if file_path:
        file_handler = logging.FileHandler(file_path, mode='w')
        file_handler.setFormatter(formater)
        logger.addHandler(file_handler)
    return logger


def get_pretrained_model_dir(name: str) -> str:
    pretrained_models = ['pl']
    if name in pretrained_models:
        root_dir = get_root_dir()
        model_dir = os.path.join(root_dir, 'deepspeech_keras\models', name)
        return model_dir
    raise ValueError('Not valid pretrained model')


def load(name: str):
    from deepspeech_keras.deepspeech import DeepSpeech
    if os.path.isdir(name):
        model_dir = name
    else:
        model_dir = get_pretrained_model_dir(name)

    config_path = os.path.join(model_dir, 'configuration.yaml')
    alphabet_path = os.path.join(model_dir, 'alphabet.txt')
    weights_path = os.path.join(model_dir, 'weights.hdf5')
    print("config path", config_path)
    assert os.path.isfile(config_path), 'The config file required in the directory'
    assert os.path.isfile(alphabet_path), 'The alphabet file required in the directory'

    deepspeech = DeepSpeech.construct(config_path, alphabet_path)
    if os.path.isfile(weights_path):
        deepspeech.load(weights_path)
    return deepspeech


def pretrained_models(func: Callable) -> Callable:

    def load_wrapper(deepspeech, file_path: str):
        if os.path.isfile(file_path):
            weights_path = file_path
        else:
            model_dir = get_pretrained_model_dir(file_path)
            weights_path = os.path.join(model_dir, 'weights.hdf5')
        return func(deepspeech, weights_path)
    return load_wrapper
