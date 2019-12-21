from typing import List, Tuple
import h5py
import numpy as np
import pandas as pd
from . import Dataset


class Features(Dataset):
    """
    The `Features` dataset keeps a reference to precomputed features and
    transcriptions in the HDF store. There is no sense to compute features
    for each epoch over and over. If you have enough disk space, this is a
    preferable approach.
    """

    def __init__(self, store: h5py.File, **kwargs):
        super().__init__(**kwargs)
        self._store = store

    @property
    def store(self):
        return self._store

    @classmethod
    def from_hdf(cls, file_path: str, **kwargs):
        """ Precomputed features and transcriptions are in the HDF file.
        Features are in HDF's datasets, stored as numpy arrays. The store
        also contains the reference table, which contains paths to features
        and transcriptions (details: https://docs.h5py.org/) """
        features_store = h5py.File(file_path, mode='r')
        references = pd.HDFStore(file_path, mode='r')['references']
        return cls(references=references, store=features_store, **kwargs)

    def get_batch(self, index: int) -> Tuple[List[np.ndarray], List[str]]:
        """ Select samples from the reference index, and read features with
        transcriptions. """
        start, end = index * self._batch_size, (index + 1) * self._batch_size
        references = self._references[start:end]
        paths, transcripts = references.path, references.transcript
        features = self._read_features(paths)
        return features, transcripts

    def _read_features(self, paths: List[str]) -> List[np.ndarray]:
        """ Read already prepared features from the store. """
        return [self._store[path][:] for path in paths]
