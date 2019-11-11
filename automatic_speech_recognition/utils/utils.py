import pickle
import logging
from functools import reduce
from logging import Logger
from typing import Any
from tensorflow import keras
import numpy as np
from scipy.io import wavfile


def load(file_path: str):
    """ Load arbitrary data from the pickled file. """
    with open(file_path, mode='rb') as file:
        return pickle.load(file)


def save(data: Any, file_path: str):
    """ Save arbitrary data in the pickled file. """
    with open(file_path, mode='wb') as file:
        pickle.dump(data, file)


def read_audio(file_path: str) -> np.ndarray:
    """ Read already prepared features from the store. """
    fs, audio = wavfile.read(file_path)
    return audio


def calculate_units(model: keras.Model) -> int:
    """ Calculate number of the model parameters. """
    units = 0
    for parameters in model.get_weights():
        units += reduce(lambda x, y: x * y, parameters.shape)
    return units


def create_logger(file_path=None, level=20, name='asr') -> Logger:
    """ Create the logger and handlers both console and file. """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s [%(levelname)-8s] [%(name)-20s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)       # handle all messages from logger (not set handler level)
    if file_path:
        file_handler = logging.FileHandler(file_path, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
