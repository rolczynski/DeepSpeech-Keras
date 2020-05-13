from typing import Collection
import tensorflow as tf
from tensorflow import keras
import logging

logger = logging.getLogger('asr.weights_load')


def load_weights(
        model: keras.Model, model_chk: str,
        skip_on_fail: bool = False, verbose=True):
    """
    Loads weights from checkpoint.
    Weights are loaded by their names which allows to have different
    architectures in current model and checkpoint.
    There is a bug in tf which prevents code from working in tf 2.0.
    More info at
    https://github.com/tensorflow/tensorflow/issues/35446.
    """
    saved_model = keras.models.load_model(model_chk, compile=False)

    # Define sets to check which weights were initialized/loaded
    skipped_layers = set()
    unused_saved_layers = {
        layer.name: layer.get_weights() for layer in saved_model.layers}

    for layer in model.layers:
        if layer.name in unused_saved_layers:
            layer_weights = unused_saved_layers[layer.name]
            layer.set_weights(layer_weights)

            del unused_saved_layers[layer.name]
        else:
            skipped_layers.add(layer.name)

    if verbose:
        _print_layer_usage_info(skipped_layers, unused_saved_layers.keys())
    if skipped_layers and not skip_on_fail:
        raise RuntimeError(
            "Runtime skipped some layers while loading model."
            " Enable verbose to know which layers."
            " were not inititalized")


def _print_layer_usage_info(skipped_layers, unused_saved_layers):
    """
    Hidden function to nicely print out which layers
    were not initialized or used in load_weights(...)
    :param skipped_layers:
    :param unused_saved_layers:
    :return:
    """
    logger.info(f"Could not load weights for"
                f" {len(skipped_layers)} layers from model.")
    for layer_name in skipped_layers:
        logger.info(layer_name)

    logger.info(f"Didn't use {len(unused_saved_layers)}"
                " weights from saved model.")
    for layer_name in unused_saved_layers:
        logger.info(layer_name)
