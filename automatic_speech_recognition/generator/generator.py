import abc
from typing import Any
import numpy as np
import pandas as pd


class Generator:
    """
    Generates data for Keras. This structure *does not* guarantee
    that the network will only train once on each sample per epoch
    which is not the case with keras.utils.Sequence. However,
    a generator can be easily processed as a stream.
    """
    def __init__(self,
                 references: pd.DataFrame,
                 batch_size=30,
                 shuffle_after_epoch=1):
        self._references = references
        self._batch_size = batch_size
        self._shuffle_after_epoch = shuffle_after_epoch
        self.epoch = 0
        self.indices = np.arange(len(self))

    def __iter__(self):
        return self.routine()

    def routine(self):
        for i in range(len(self)):
            yield self[i]
        self.epoch += 1         # On epoch end
        self._shuffle_indices()

    def __len__(self) -> int:
        """ Denotes the number of batches per epoch. """
        return int(np.floor(len(self._references.index) / self._batch_size))

    def __getitem__(self, index: int) -> Any:
        """ Operator to get the batch data. """
        batch_index = self.indices[index]
        return self._get_batch(batch_index)

    @abc.abstractmethod
    def _get_batch(self, index: int) -> Any:
        pass

    def _shuffle_indices(self):
        """ Set up the order of next batches """
        if self.epoch >= self._shuffle_after_epoch:
            np.random.shuffle(self.indices)
