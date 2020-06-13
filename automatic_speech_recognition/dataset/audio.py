from typing import Tuple, List
import numpy as np
import pandas as pd
import os
from . import Dataset
from .. import utils


class Audio(Dataset):
    """
    The `Audio` dataset keeps a reference to audio files and corresponding
    transcriptions. The audio files are read and then return with
    transcriptions. Now, we support only csv files.
    """
    def __init__(self, use_filesizes=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sr = None
        self.use_filesizes = use_filesizes


    @classmethod
    def from_csv(cls, file_path: str,
                 use_filesizes=True, **kwargs):
        """ The reference csv file contains paths and transcripts,
        which are comma separated.
        :param use_filesizes: try to read file sizes from csv
        """
        cols = ['path', 'transcript']
        if use_filesizes:
            cols.append('filesize')
        references = pd.read_csv(file_path, usecols=cols,
                                 sep=',', encoding='utf-8', header=0)

        return cls(references=references, use_filesizes=use_filesizes,
                   **kwargs)

    def get_batch(self, index: int) -> Tuple[List[np.ndarray], List[str]]:
        """ Select samples from the reference index, read audio files and
        return with transcriptions. """
        start, end = index * self._batch_size, (index + 1) * self._batch_size
        references = self._references[start:end]
        paths, transcripts = (references.path,
                              references.transcript.tolist())
        batch_audio_data = [utils.read_audio(file_path) for file_path
                            in paths]

        batch_srs = [data[1] for data in batch_audio_data]
        batch_audio = [data[0] for data in batch_audio_data]

        if self.sr is None:
            self.sr = batch_srs[0]
        assert np.all(np.equal(batch_srs, self.sr)), "Not all sample rates are equal"
        return batch_audio, transcripts

    def sort_by_length(self):
        if self.use_filesizes:
            self._references.sort_values('filesize', inplace=True)
        else:
            self._references['lengths'] = self._references.transcript.str.len()
            self._references.sort_values('lengths', inplace=True)
