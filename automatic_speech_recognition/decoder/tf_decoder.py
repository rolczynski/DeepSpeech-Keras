from typing import List
import numpy as np
import tensorflow as tf
from .decoder import Decoder


class TensorflowDecoder(Decoder):

    def __init__(self, beam_size: int):
        self.beam_size = beam_size

    def __call__(self, batch_probs: np.ndarray) -> List[np.ndarray]:
        batch_size, sequence_length, features_num = batch_probs.shape
        sequence_lengths = [sequence_length] * batch_size
        (top_paths, ), ranks = tf.keras.backend.ctc_decode(batch_probs, sequence_lengths, top_paths=1,
                                                           greedy=False, beam_width=self.beam_size)
        return top_paths
