import os
import pickle
import logging
from functools import reduce
from logging import Logger
from typing import Any
import numpy as np
from scipy.io import wavfile
from tensorflow import keras
from google.cloud import storage

logger = logging.getLogger('asr.utils')


def load(file_path: str):
    """ Load arbitrary python objects from the pickled file. """
    with open(file_path, mode='rb') as file:
        return pickle.load(file)


def save(data: Any, file_path: str):
    """ Save arbitrary python objects in the pickled file. """
    with open(file_path, mode='wb') as file:
        pickle.dump(data, file)


def download_from_bucket(bucket_name: str, remote_path: str, local_path: str):
    """ Download the file from the public bucket. """
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blob = storage.Blob(remote_path, bucket)
    blob.download_to_filename(local_path, client=client)


def maybe_download_from_bucket(bucket_name: str, remote_path: str, local_path: str):
    """ Download file from the bucket if it does not exist. """
    if os.path.isfile(local_path):
        return
    directory = os.path.dirname(local_path)
    os.makedirs(directory, exist_ok=True)
    logger.info('Downloading file from the bucket...')
    download_from_bucket(bucket_name, remote_path, local_path)


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
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-8s] [%(name)-20s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)  # handle all messages from logger
    if file_path:
        file_handler = logging.FileHandler(file_path, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
