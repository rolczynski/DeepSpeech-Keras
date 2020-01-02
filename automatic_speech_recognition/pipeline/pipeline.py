import abc
from typing import List
import numpy as np
from tensorflow import keras
from .. import decoder
from .. import features
from .. import dataset
from .. import text


class Pipeline:

    @property
    @abc.abstractmethod
    def alphabet(self) -> text.Alphabet:
        pass

    @property
    @abc.abstractmethod
    def features_extractor(self) -> features.FeaturesExtractor:
        pass

    @property
    @abc.abstractmethod
    def model(self) -> keras.Model:
        pass

    @property
    @abc.abstractmethod
    def decoder(self) -> decoder.Decoder:
        pass

    @abc.abstractmethod
    def fit(self,
            train_source: dataset.Dataset,
            dev_source: dataset.Dataset,
            prepared_features=False,
            **kwargs) -> keras.callbacks.History:
        pass

    @abc.abstractmethod
    def predict(self, batch_audio: List[np.ndarray],  **kwargs) -> List[str]:
        pass

    @abc.abstractmethod
    def save(self, directory: str):
        pass
