import abc
from typing import List, Tuple
import numpy as np
import pandas as pd
from tensorflow import keras


class Dataset(keras.utils.Sequence):
    """
    The `Dataset` represents the sequence of samples used for Keras models.
    It has a view by the `reference` to sample sources, so we do not keep an
    entire dataset in the memory.

    The class contains two essential methods `len` and `getitem`, which are
    required to use the `keras.utils.Sequence` interface. This structure
    guarantee that the network only trains once on each sample per epoch.
    """

    def __init__(self,
                 references: pd.DataFrame,
                 batch_size: int):
        self._batch_size = batch_size
        self._references = references
        self._indices = np.arange(len(self))

    @property
    def indices(self):
        return self._indices

    def __len__(self) -> int:
        """ Indicate the number of batches per epoch. """
        return int(np.floor(len(self._references.index) / self._batch_size))

    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], List[str]]:
        """ Get the batch data. We have an auxiliary index to have more
        control of the order, because basically model uses it sequentially. """
        aux_index = self._indices[index]
        return self.get_batch(aux_index)

    @abc.abstractmethod
    def get_batch(self, index: int) -> Tuple[List[np.ndarray], List[str]]:
        pass

    def shuffle_indices(self):
        """ Set up the order of return batches. """
        np.random.shuffle(self._indices)
