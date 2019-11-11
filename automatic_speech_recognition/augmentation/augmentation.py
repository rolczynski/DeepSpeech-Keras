import abc
import numpy as np


class Augmentation:

    @abc.abstractmethod
    def __call__(self, batch_features: np.ndarray) -> np.ndarray:
        pass
