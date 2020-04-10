import os
from types import MethodType
import logging
from typing import List, Callable, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from . import Pipeline
from .. import augmentation
from .. import decoder
from .. import features
from .. import dataset
from .. import text
from .. import utils
from ..features import FeaturesExtractor

logger = logging.getLogger('asr.pipeline')


class CTCPipeline(Pipeline):
    """
    The pipeline is responsible for connecting a neural network model with
    all non-differential transformations (features extraction or decoding),
    and dependencies. Components are independent.
    """

    def __init__(self,
                 alphabet: text.Alphabet,
                 features_extractor: features.FeaturesExtractor,
                 model: keras.Model,
                 optimizer: keras.optimizers.Optimizer,
                 decoder: decoder.Decoder,
                 gpus: List[str] = None):
        self._alphabet = alphabet
        self._model_cpu = model
        self._optimizer = optimizer
        self._decoder = decoder
        self._features_extractor = features_extractor
        self._gpus = gpus
        self._model = self.distribute_model(model, gpus) if gpus else model

    @property
    def alphabet(self) -> text.Alphabet:
        return self._alphabet

    @property
    def features_extractor(self) -> features.FeaturesExtractor:
        return self._features_extractor

    @property
    def model(self) -> keras.Model:
        return self._model_cpu

    @property
    def decoder(self) -> decoder.Decoder:
        return self._decoder

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
        return (features, feature_lengths, label_lengths), labels

    def compile_model(self):
        """ The compiled model means the model configured for training. """
        y = keras.layers.Input(name='y', shape=[None], dtype='int32')
        loss = self.get_loss()
        self._model.compile(self._optimizer, loss, target_tensors=[y])
        logger.info("Model is successfully compiled")

    def fit(self,
            dataset: dataset.Dataset,
            dev_dataset: dataset.Dataset,
            augmentation: augmentation.Augmentation = None,
            prepared_features: bool = False,
            **kwargs) -> keras.callbacks.History:
        """ Get ready data, compile and train a model. """
        dataset = self.wrap_preprocess(dataset, prepared_features, augmentation)
        dev_dataset = self.wrap_preprocess(dev_dataset, prepared_features, augmentation)
        if not self._model.optimizer:  # a loss function and an optimizer
            self.compile_model()  # have to be set before the training
        return self._model.fit(dataset, validation_data=dev_dataset, **kwargs)

    def predict(self, batch_audio: List[np.ndarray], **kwargs) -> List[str]:
        """ Get ready features, and make a prediction. """
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

    def wrap_preprocess(self,
                        dataset: dataset.Dataset,
                        is_extracted: bool,
                        augmentation: augmentation.Augmentation):
        """ Dataset does not know the feature extraction process by design.
        The Pipeline class exclusively understand dependencies between
        components. """

        def preprocess(get_batch):
            def get_prep_batch(index: int):
                batch = get_batch(index)
                return self.preprocess(batch, is_extracted, augmentation)

            return get_prep_batch

        dataset.get_batch = preprocess(dataset.get_batch)
        return dataset

    def save(self, directory: str):
        """ Save each component of the CTC pipeline. """
        self._model.save(os.path.join(directory, 'model.tf'))
        utils.save(self._alphabet, os.path.join(directory, 'alphabet.bin'))
        utils.save(self._decoder, os.path.join(directory, 'decoder.bin'))
        utils.save(self._features_extractor,
                   os.path.join(directory, 'feature_extractor.bin'))

    @classmethod
    def load(cls, directory: str, **kwargs):
        """ Load each component of the CTC pipeline. """
        model = keras.model.load_model(os.path.join(directory, 'model.tf'))
        alphabet = utils.load(os.path.join(directory, 'alphabet.bin'))
        decoder = utils.load(os.path.join(directory, 'decoder.bin'))
        features_extractor = utils.load(
            os.path.join(directory, 'feature_extractor.bin'))
        return cls(alphabet, model, model.optimizer, decoder,
                   features_extractor, **kwargs)

    @staticmethod
    def distribute_model(model: keras.Model, gpus: List[str]) -> keras.Model:
        """ Replicates a model on different GPUs. """
        try:
            dist_model = keras.utils.multi_gpu_model(model, len(gpus))
            logger.info("Training using multiple GPUs")
        except ValueError:
            dist_model = model
            logger.info("Training using single GPU or CPU")
        return dist_model

    def get_loss(self) -> Callable:
        """ The CTC loss using TensorFlow's `ctc_loss`. """
        def ctc_loss(labels, logits):
            label_lenths = keras.backend.print_tensor(self.model.inputs[2][:, 0], 'label_lengths')
            logit_lengths = keras.backend.print_tensor(self.model.inputs[1][:, 0], 'logit_lengths')
            labels_ = keras.backend.print_tensor(labels, 'labels')
            logits_ = keras.backend.print_tensor(logits, 'logits')
            return tf.nn.ctc_loss(labels_, logits_, label_lenths,
                                  logit_lengths,
                                  logits_time_major=False, blank_index=-1)
        return ctc_loss
