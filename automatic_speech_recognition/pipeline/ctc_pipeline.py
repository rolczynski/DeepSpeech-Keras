import os
import logging
from typing import Generator, List, Callable
import numpy as np
import tensorflow as tf
from tensorflow import keras
from . import Pipeline
from .. import augmentation
from .. import decoder
from .. import features
from .. import generator
from .. import text
from .. import utils
logger = logging.getLogger('asr.pipeline')


class CTCPipeline(Pipeline):
    def __init__(self,
                 alphabet: text.Alphabet,
                 model: keras.Model,
                 optimizer: keras.optimizers.Optimizer,
                 decoder: decoder.Decoder,
                 features_extractor: features.FeaturesExtractor = None,
                 gpus: List[str] = None):
        self._alphabet = alphabet
        self._model = model
        self._optimizer = optimizer
        self._decoder = decoder
        self._features_extractor = features_extractor
        self._gpus = gpus
        self._compiled_model = self.distribute_model(model, gpus) if gpus else model

    @property
    def alphabet(self) -> text.Alphabet:
        return self._alphabet

    @property
    def features_extractor(self) -> features.FeaturesExtractor:
        return self._features_extractor

    @property
    def model(self) -> keras.Model:
        return self._model

    @property
    def decoder(self) -> decoder.Decoder:
        return self._decoder

    def fit(self,
            train_data: generator.Generator,
            dev_data: generator.Generator,
            augmentation: augmentation.Augmentation = None,
            prepared_features: bool = False,
            **kwargs) -> keras.callbacks.History:
        """ Get ready data, compile and train a model. """
        train_gen = self._preprocess_data(train_data, prepared_features, augmentation)
        dev_gen = self._preprocess_data(dev_data, prepared_features)
        if not self._compiled_model.is_compiled:        # A loss function and an optimizer have
            self.compile_model(self._optimizer)         # to be set before the training.
        return self._compiled_model.fit_generator(generator=train_gen, validation_data=dev_gen, **kwargs)

    def predict(self,
                batch_audio: List[np.ndarray] = None,
                features: np.ndarray = None,
                **kwargs) -> List[str]:
        """ Get ready features, and make a prediction. Features can be extracted
        outside of this method, however, it is not recommended. """
        features = self._features_extractor(batch_audio) if not features else features
        batch_probs = self._compiled_model.predict(features, **kwargs)
        decoded_labels = self._decoder(batch_probs)
        predictions = self._alphabet.get_batch_transcripts(decoded_labels)
        return predictions

    def _preprocess_data(self,
                         data: generator.Generator,
                         prepared_features: bool = False,
                         augmentation: augmentation.Augmentation = None) -> Generator:
        """ Enrich generator by extracting features, and converting transcripts to labels.
        Do an augmentation if it is needed. """
        features = self._extract_features(data) if not prepared_features else data
        gen = self._convert_transcript_to_labels(features)
        return gen if not augmentation else self.augment_features(gen, augmentation)

    def _extract_features(self, generator: Generator) -> Generator:
        return ([self._features_extractor(batch_audio), transcripts] for batch_audio, transcripts in generator)

    def _convert_transcript_to_labels(self, generator: Generator) -> Generator:
        return ([features, self._alphabet.get_batch_labels(transcripts)] for features, transcripts in generator)

    def compile_model(self, optimizer: keras.optimizers.Optimizer) -> None:
        """ The compiled model means the model configured for training. """
        y = keras.layers.Input(name='y', shape=[None], dtype='int32')
        loss = self.get_loss()
        self._compiled_model.compile(optimizer, loss, target_tensors=[y])
        logger.info("Model is successfully compiled")

    def save(self, directory: str):
        """ Save each component of the CTC pipeline. """
        self._model.save(os.path.join(directory, 'model.h5'))
        utils.save(self._alphabet, os.path.join(directory, 'alphabet.bin'))
        utils.save(self._decoder, os.path.join(directory, 'decoder.bin'))
        utils.save(self._features_extractor, os.path.join(directory, 'feature_extractor.bin'))

    @classmethod
    def load(cls, directory: str, **kwargs):
        """ Load each component of the CTC pipeline. """
        model = keras.model.load_model(os.path.join(directory, 'model.h5'))
        alphabet = utils.load(os.path.join(directory, 'alphabet.bin'))
        decoder = utils.load(os.path.join(directory, 'decoder.bin'))
        features_extractor = utils.load(os.path.join(directory, 'feature_extractor.bin'))
        return cls(alphabet, model, model.optimizer, decoder, features_extractor, **kwargs)

    @staticmethod
    def augment_features(generator: Generator, augmentation: augmentation.Augmentation) -> Generator:
        return ([augmentation(features), labels] for features, labels in generator)

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

    @staticmethod
    def get_loss() -> Callable:
        """ The CTC loss using TensorFlow's `ctc_loss` using Keras backend. """
        def get_length(tensor):
            lengths = tf.math.reduce_sum(tf.ones_like(tensor), 1)
            return tf.reshape(tf.cast(lengths, tf.int32), [-1, 1])

        def ctc_loss(y, y_hat):
            sequence_length = get_length(tf.math.reduce_max(y_hat, 2))
            label_length = get_length(y)
            return tf.nn.ctc_loss(y, y_hat, sequence_length, label_length)
        return ctc_loss
