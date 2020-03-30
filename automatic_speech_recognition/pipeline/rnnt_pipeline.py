import os
from types import MethodType
import logging
from typing import List, Callable, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
import collections.abc
from . import Pipeline
from .. import augmentation
from .. import decoder
from .. import features
from .. import dataset
from .. import text
from .. import utils
from . import CTCPipeline
from ..features import FeaturesExtractor

logger = logging.getLogger('asr.pipeline')

try:
    from warprnnt_tensorflow import rnnt_loss
except:
    logger.info("Could not import warp-rnnt loss")


class RNNTPipeline(CTCPipeline):
    """
    Pipeline modifies preprocessing step to feed labels as model inputs as well.
    """

    def preprocess(self,
                   batch: Tuple[List[np.ndarray], List[str]],
                   is_extracted: bool,
                   augmentation: augmentation.Augmentation):
        """ Preprocess batch data to format understandable to a model. """
        data, transcripts = batch
        if is_extracted:  # then just align features
            feature_lengths = [len(feature_seq) for feature_seq in data]
            features = FeaturesExtractor.align(data)
        else:
            features, feature_lengths = self._features_extractor(data, return_lengths=True)
        features = augmentation(features) if augmentation else features
        feature_lengths = np.array(feature_lengths)

        labels = self._alphabet.get_batch_labels(transcripts)
        label_lengths = np.array([len(decoded_text) for decoded_text in batch[1]])
        return (features, labels, feature_lengths, label_lengths), labels

    def predict(self, batch_audio: List[np.ndarray], **kwargs) -> List[str]:
        """ Get ready features, and make a prediction. All additional model inputs
        except for the first one are replaced by zero tensors"""
        # TODO Found implementation on https://github.com/zhanghaobaba/RNN-Transducer/blob/786fa75ff65c8ce859183d3c67aa408ff7fdef13/model.py#L33
        # need to tansfer it to tensorflow
        raise NotImplementedError("prediction is not yet implemented")
        features = self._features_extractor(batch_audio)

        # Make prediction with substituted additional tensors
        input_tensors = [features]
        for inp in self._model.inputs[1:]:
            place_holder_shape = [dim_size if dim_size is not None else 0 for dim_size in inp.shape]
            input_tensors.append(tf.zeros(place_holder_shape))
        batch_logits = self._model.predict(input_tensors, **kwargs)

        decoded_labels = self._decoder(batch_logits)
        predictions = self._alphabet.get_batch_transcripts(decoded_labels)
        return predictions

    def get_loss(self) -> Callable:
        """ The CTC loss using TensorFlow's `ctc_loss`. """

        def rnnt_loss_wrapper(labels, outputs):
            return rnnt_loss(outputs,
                             labels,
                             self.model.inputs[2][:, 0],
                             self.model.inputs[3][:, 0],
                             blank_label=self._alphabet.blank_token)

        return rnnt_loss_wrapper
