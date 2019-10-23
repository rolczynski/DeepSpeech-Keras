import numpy as np
from typing import List
import math
import collections
import tensorflow as tf
from typing import Callable
from keras.backend import tensorflow_backend as K

from deepspeech_keras.text import Alphabet
NEG_INF = -float("inf")


def batch_tensorflow_decode(y_hat, decoder: Callable, alphabet: Alphabet):
    """ Enable to batch decode using tensorflow decoder. """
    labels, = decoder([y_hat])
    return alphabet.get_batch_transcripts(labels)


def get_tensorflow_decoder(output_tensor, beam_size=1024):
    """ The TensorFlow implementation of the CTC decoder. """
    def get_length(tensor):
        lengths = tf.reduce_sum(tf.ones_like(tensor), 1)
        return tf.cast(lengths, tf.int32)

    sequence_length = get_length(tf.reduce_max(output_tensor, 2))
    top_k_decoded, _ = K.ctc_decode(output_tensor, sequence_length, greedy=False, beam_width=beam_size)
    decoder = K.function([output_tensor], [top_k_decoded[0]])
    return decoder


def batch_naive_decode(batch_y_hat, alphabet, **kwargs):
    """ Enable to batch decode (using multiprocessing in the future). """
    return [naive_decode(y_hat, alphabet, **kwargs) for y_hat in batch_y_hat]


def naive_decode(probs: np.ndarray, alphabet: Alphabet, beam_size=1024, prune=0.001):
    """
    Performs inference for the given output probabilities.
    Arguments:
        probs: The output probabilities (e.g. post-softmax) for each
          time step. Should be an array of shape (time x output dim).
        beam_size (int): Size of the beam to use during inference.

    Returns the output label sequence and the corresponding negative
    log-likelihood estimated by the decoder.

    For more details checkout either of these references:
        https://distill.pub/2017/ctc/#inference
        https://arxiv.org/abs/1408.2873
    """
    Time, _ = probs.shape
    probs, prune = np.log(probs), np.log(prune)
    beam = [('', (0.0, NEG_INF))]

    for time in range(Time):
        next_beam = make_beam()

        pruned_indices, = np.where(probs[time] > prune)
        probable_characters = [(i, alphabet.string_from_label(i)) for i in pruned_indices]

        for char_index, char in probable_characters:
            char_prob = probs[time, char_index]

            for prefix, (P_b, P_nb) in beam:
                update_next_beam(next_beam, prefix, char, char_prob, [P_b, P_nb])

        beam = choose_best_candidates(next_beam, n=beam_size)
        # rescore(beam)

    best = beam[0]
    return best[0], -logsum(*best[1])


def update_next_beam(next_beam: collections.defaultdict,
                     prefix: str, char: str, char_prob: int, probs: List):
    """ Implementation CTC logic for calculation prefix probabilities. """
    P, [P_b, P_nb] = char_prob, probs
    prefix_end = prefix[-1] if prefix else None
    next_prefix = prefix + char

    if char == '':
        # If we propose a blank the prefix doesn't change.
        # Only the probability of ending in blank gets updated.
        next_P_b, next_P_nb = next_beam[prefix]
        next_P_b = logsum(next_P_b, P_b + P, P_nb + P)
        next_beam[prefix] = (next_P_b, next_P_nb)
        return

    elif char == prefix_end:
        # We don't include the previous probability of not ending
        # in blank (P_nb) if s is repeated at the end. The CTC
        # algorithm merges characters not separated by a blank.
        next_P_b, next_P_nb = next_beam[next_prefix]
        next_P_nb = logsum(next_P_nb, P_b + P)
        next_beam[next_prefix] = (next_P_b, next_P_nb)

        # Also update the unchanged prefix. This is the merging case.
        next_P_b, next_P_nb = next_beam[prefix]
        next_P_nb = logsum(next_P_nb, P_nb + P)
        next_beam[prefix] = (next_P_b, next_P_nb)
        return

    else:
        # Extend the prefix by the new character s and add it to
        # the beam. Only the probability of not ending in blank
        # gets updated.
        next_P_b, next_P_nb = next_beam[next_prefix]
        next_P_nb = logsum(next_P_nb, P_b + P, P_nb + P)
        next_beam[next_prefix] = (next_P_b, next_P_nb)
        return


def choose_best_candidates(beam: collections.defaultdict, n: int):
    """ Sort and trim the beam before moving on to the next search."""
    next_beam = sorted(beam.items(), key=lambda x: logsum(*x[1]), reverse=True)
    return next_beam[:n]


def make_beam():
    """ A default dictionary to store the next step candidates. """
    return collections.defaultdict(lambda: (NEG_INF, NEG_INF))


def logsum(*args):
    """ Compute log sum using exp trick. """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


def rescore(beam: collections.defaultdict):
    """
    Take a use of misspelling rules (the confusion matrix) to put the performance up.

    References:
        - https://github.com/barrust/pyspellchecker
    """
    raise NotImplemented
