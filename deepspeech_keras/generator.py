from typing import List

import h5py
import numpy as np
import pandas as pd
from keras.utils import Sequence
from deepspeech_keras.text import Alphabet
from deepspeech_keras.audio import FeaturesExtractor
from deepspeech_keras.augmentation import mask_features


class DataGenerator(Sequence):
    """
    Generates data for Keras

    `Sequence` are a safer way to do multiprocessing. This structure
    guarantees that the network will only train once on each sample per epoch
    which is not the case with generators.

    References:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    """

    def __init__(self,
                 references: pd.DataFrame,
                 alphabet: Alphabet,
                 features_extractor: FeaturesExtractor,
                 shuffle_after_epoch=1,
                 batch_size=30,
                 features_store=False,
                 mask=False,
                 mask_params=None):
        self._references = references
        self._features_store = features_store
        self._alphabet = alphabet
        self._features_extractor = features_extractor
        self._batch_size = batch_size
        self._shuffle_after_epoch = shuffle_after_epoch
        self.epoch = 0
        self.indices = np.arange(len(self))
        self.mask = mask
        self.mask_params = mask_params

    @classmethod
    def from_audio_files(cls, file_path, **kwargs):
        """ Create generator from csv file. The file contains audio file paths
        with corresponding transcriptions. """
        references = pd.read_csv(file_path, usecols=['path', 'transcript'], sep=',', encoding='utf-8', header=0)
        return cls(references, **kwargs)

    @classmethod
    def from_prepared_features(cls, file_path, **kwargs):
        """ Create generator from prepared features saved in the HDF5 format.
        The hdf5 file has the hierarchy with /-separator and also can be invoke via `path`. """
        features_store = h5py.File(file_path, mode='r')
        references = pd.HDFStore(file_path, mode='r')['references']  # Read DataFrame via PyTables
        return cls(references, features_store=features_store, **kwargs)

    def __len__(self):
        """ Denotes the number of batches per epoch. """
        return int(np.floor(len(self._references.index) / self._batch_size))

    def __getitem__(self, index):
        """ Operator to get the batch data. """
        batch_index = self.indices[index]
        return self._get_batch(batch_index)

    def _get_batch(self, index):
        """ Read (if features store exist) or generate features and labels batch. """
        start, end = index * self._batch_size, (index + 1) * self._batch_size
        references = self._references[start:end]
        paths, transcripts = references.path, references.transcript

        labels = self._alphabet.get_batch_labels(transcripts)
        if self._features_store:
            features = self._read_features(paths)
        else:
            features = self._extract_features(paths)

        if self.mask:
            features = self._mask_features(features)

        return features, labels

    def _read_features(self, paths):
        """ Read already prepared features from the store. """
        features = [self._features_store[path][:] for path in paths]
        return self._features_extractor.align(features)

    def _extract_features(self, paths):
        """ Extract features from the audio files (mono 16kHz). """
        return self._features_extractor.get_features(files=paths)

    def _mask_features(self, features):
        """ SpecAugment: A Simple Data Augmentation Method. """
        return np.stack([mask_features(sample, **self.mask_params) for sample in features], axis=0)

    def on_epoch_end(self):
        """ Invoke methods at the end of the each epoch. The fit method should have: `shuffle=False`.
        Keras OrderedEnqueuer seems to run on async on two threads so the epoch number is counted twice (bug). """
        self.epoch += 1
        self._shuffle_indices()

    def _shuffle_indices(self):
        """ Set up the order of next batches """
        if self.epoch >= self._shuffle_after_epoch:
            np.random.shuffle(self.indices)


class DistributedDataGenerator(Sequence):

    def __init__(self, generators: List[DataGenerator]):
        self._generators = generators
        self._generator_sizes = [len(generator) for generator in self._generators]
        self._generator_limits = np.cumsum(self._generator_sizes)
        self.epoch = 0
        self.indices = np.arange(len(self))
        np.random.shuffle(self.indices)

    @classmethod
    def from_audio_files(cls, file_paths: List[str], **kwargs):
        generators = []
        for file_path in file_paths:
            generator = DataGenerator.from_audio_files(file_path, **kwargs)
            generators.append(generator)
        return cls(generators)

    @classmethod
    def from_prepared_features(cls, file_paths: List[str], **kwargs):
        generators = []
        for file_path in file_paths:
            generator = DataGenerator.from_prepared_features(file_path, **kwargs)
            generators.append(generator)
        return cls(generators)

    def __len__(self):
        """ Denotes the number of batches per epoch. """
        return sum(self._generator_sizes)

    def __getitem__(self, index):
        """ Operator to get the batch data. """
        next_index = self.indices[index]
        generator_index = np.searchsorted(self._generator_limits, next_index, side='right')     # Right side because limits are based on the lengths, not indices
        generator = self._generators[generator_index]
        if generator_index == 0:
            gen_index = next_index
        else:
            gen_index = next_index - self._generator_limits[generator_index-1]
        return generator[gen_index]

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        self.epoch += 1
        for generator in self._generators:
            generator.on_epoch_end()
