import functools
import operator
from typing import Callable, List, Union, Tuple
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from . import evaluate
from .. import dataset
from .. import pipeline


def save_metrics_and_activations(pipeline: pipeline.Pipeline,
                                 dataset: dataset.Dataset,
                                 store_path: str,
                                 prepared_features: bool = False,
                                 return_metrics: bool = False
                                 ) -> Union[Tuple[float, float], pd.DataFrame]:

    columns = ['sample_id', 'transcript', 'prediction', 'wer', 'cer']
    references = pd.DataFrame(columns=columns).set_index('sample_id')
    get_activations = get_activations_function(pipeline.model)

    with h5py.File(store_path, mode='w') as store:
        for data, transcripts in dataset:
            features = data if prepared_features else pipeline.features_extractor(data)
            *activations, y_hat = get_activations([features, 0])
            decoded_labels = pipeline.decoder(y_hat)
            predictions = pipeline.alphabet.get_batch_transcripts(decoded_labels)
            batch_metrics = list(evaluate.get_metrics(sources=predictions,
                                                      destinations=transcripts))

            save_in_store(store, [*activations, y_hat], batch_metrics, references)

    with pd.HDFStore(store_path, mode='r+') as store:
        store.put('references', references)
    metrics = pd.DataFrame(functools.reduce(operator.concat, batch_metrics))
    return metrics if return_metrics else (metrics.wer.mean(), metrics.cer.mean())


def get_activations_function(model: keras.Model) -> Callable:
    """ Function which handle all activations through one pass. """
    inputs = [model.input, tf.keras.learning_phase()]
    outputs = [layer.output for layer in model.layers][1:]
    return tf.keras.function(inputs, outputs)


def save_in_store(store: h5py.File,
                  layer_outputs: List[np.ndarray],
                  metrics: List[evaluate.Metric],
                  references: pd.DataFrame):
    """ Save batch data into HDF5 file. """
    for index, metric in enumerate(metrics):
        sample_id = len(references)
        references.loc[sample_id] = metric
        for output_index, batch_layer_outputs in enumerate(layer_outputs):
            layer_output = batch_layer_outputs[index]
            store.create_dataset(f'outputs/{output_index}/{sample_id}', data=layer_output)
