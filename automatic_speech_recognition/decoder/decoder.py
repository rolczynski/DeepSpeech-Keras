import abc
from typing import List
import numpy as np


class Decoder:

    @abc.abstractmethod
    def __call__(self, batch_probs: np.ndarray) -> List[np.ndarray]:
        pass
