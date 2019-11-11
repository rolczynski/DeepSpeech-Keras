from typing import List, Tuple
import h5py
import numpy as np
import pandas as pd
from . import Generator
from .. import features


class FeaturesGenerator(Generator):

    def __init__(self, features_store: h5py.File, **kwargs):
        super().__init__(**kwargs)
        self.features_store = features_store

    @classmethod
    def from_hdf(cls, file_path: str, **kwargs):
        """ Create generator from prepared features saved in the HDF5 format.
        The hdf5 file has the hierarchy with /-separator and also can be invoke via `path`. """
        features_store = h5py.File(file_path, mode='r')
        references = pd.HDFStore(file_path, mode='r')['references']  # Read DataFrame via PyTables
        return cls(references, features_store=features_store, **kwargs)

    def _get_batch(self, index: int) -> Tuple[np.ndarray, List[str]]:
        """ Read (if features store exist) or generate features and labels batch. """
        start, end = index * self._batch_size, (index + 1) * self._batch_size
        references = self._references[start:end]
        paths, transcripts = references.path, references.transcript
        features = self._read_features(paths)
        return features, transcripts

    def _read_features(self, paths: List[str]) -> np.ndarray:
        """ Read already prepared features from the store. """
        X = [self._features_store[path][:] for path in paths]
        return features.FeaturesExtractor.align(X)
