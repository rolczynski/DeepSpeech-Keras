from typing import Tuple, List
import numpy as np
import pandas as pd
from . import Generator
from .. import utils


class DataGenerator(Generator):

    @classmethod
    def from_csv(cls, file_path: str, **kwargs):
        """ Create generator from csv file. The file contains features file paths
        with corresponding transcriptions. """
        references = pd.read_csv(file_path, usecols=['path', 'transcript'], sep=',', encoding='utf-8', header=0)
        return cls(references, **kwargs)

    def _get_batch(self, index: int) -> Tuple[List[np.ndarray], List[str]]:
        """ Read (if features store exist) or generate features and labels batch. """
        start, end = index * self._batch_size, (index + 1) * self._batch_size
        references = self._references[start:end]
        paths, transcripts = references.path, references.transcript
        batch_audio = [utils.read_audio(file_path) for file_path in paths]
        return batch_audio, transcripts
