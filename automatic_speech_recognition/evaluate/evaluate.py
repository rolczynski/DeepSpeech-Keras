from typing import List, Iterable, Tuple, Union
from collections import namedtuple
import pandas as pd
from . import distance
from .. import dataset
from .. import pipeline
Metric = namedtuple('Metric', ['transcript', 'prediction', 'wer', 'cer'])


def calculate_error_rates(pipeline: pipeline.Pipeline,
                          dataset: dataset.Dataset,
                          prepared_features: bool = False,
                          return_metrics: bool = False
                          ) -> Union[Tuple[float, float], pd.DataFrame]:
    """ Calculate base metrics: WER and CER. """
    metrics = []
    for data, transcripts in dataset:
        if prepared_features:
            predictions = pipeline.predict(data)
        else:
            predictions = pipeline.predict(batch_audio=data)
        batch_metrics = get_metrics(sources=predictions,
                                    destinations=transcripts)
        metrics.extend(batch_metrics)
    metrics = pd.DataFrame(metrics)
    return metrics if return_metrics else (metrics.wer.mean(), metrics.cer.mean())


def get_metrics(sources: List[str],
                destinations: List[str]) -> Iterable[Metric]:
    """ Calculate base metrics in one batch: WER and CER. """
    for source, destination in zip(sources, destinations):
        wer_distance, *_ = distance.edit_distance(source.split(),
                                                  destination.split())
        wer = wer_distance / len(destination.split())

        cer_distance, *_ = distance.edit_distance(list(source),
                                                  list(destination))
        cer = cer_distance / len(destination)
        yield Metric(destination, source, wer, cer)
