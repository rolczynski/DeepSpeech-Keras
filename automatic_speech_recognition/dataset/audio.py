from typing import Tuple, List
import numpy as np
import pandas as pd
from . import Dataset
from .. import utils


class Audio(Dataset):
    """
    The `Audio` dataset keeps a reference to audio files and corresponding
    transcriptions. The audio files are read and then return with
    transcriptions. Now, we support only csv files.
    """

    @classmethod
    def from_csv(cls, file_path: str, **kwargs):
        """ The reference csv file contains paths and transcripts, which are
        comma separated. """
        references = pd.read_csv(file_path, usecols=['path', 'transcript'],
                                 sep=',', encoding='utf-8', header=0)
        return cls(references=references, **kwargs)

    def get_batch(self, index: int) -> Tuple[List[np.ndarray], List[str]]:
        """ Select samples from the reference index, read audio files and
        return with transcriptions. """
        start, end = index * self._batch_size, (index + 1) * self._batch_size
        references = self._references[start:end]
        paths, transcripts = references.path, references.transcript.tolist()
        batch_audio = [utils.read_audio(file_path) for file_path in paths]
        return batch_audio, transcripts
