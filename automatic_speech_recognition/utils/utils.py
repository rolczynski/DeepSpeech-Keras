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
import tensorflow as tf
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile

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


def read_audio(file_path: str):
    """ Read already prepared features from the store. """
    audio = tf.io.read_file(file_path)
    waveform = tf.audio.decode_wav(audio)
    return 16000, waveform


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


def load_deepspeech_graph(graph_path):
    # GRAPH_PB_PATH = '/content/deepspeech-0.6.1-models/output_graph.pb'
    gr = tf.Graph()
    wts = {}
    with tf.compat.v1.Session(graph=gr) as sess:
        print("load graph")
        with gfile.FastGFile(graph_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='', )
        graph_nodes = [n for n in graph_def.node]
        names = []
        for t in graph_nodes:
            names.append(t.name)
        print(names)
        all_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        print(all_vars)
        for n in graph_nodes:
            if n.op == "Const":
                wts[n.name] = tensor_util.MakeNdarray(n.attr["value"].tensor)
    return wts, gr
