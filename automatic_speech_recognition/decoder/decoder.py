import abc
import itertools
from typing import List
import numpy as np


class Decoder:

    @abc.abstractmethod
    def __call__(self, batch_logits: np.ndarray) -> List[np.ndarray]:
        pass


class GreedyDecoder:

    def __call__(self, batch_logits: np.ndarray) -> List[np.ndarray]:
        """ Decode the best guess from logits using greedy algorithm. """
        # Choose the class with maximum probability
        best_candidates = np.argmax(batch_logits, axis=2)
        # Merge repeated chars
        decoded = [np.array([k for k, _ in itertools.groupby(best_candidate)])
                   for best_candidate in best_candidates]
        return decoded
