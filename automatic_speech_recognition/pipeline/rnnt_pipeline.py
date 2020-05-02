import os
from types import MethodType
import logging
from typing import List, Callable, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
import collections.abc
from . import Pipeline
from .. import augmentation
from . import CTCPipeline
from ..features import FeaturesExtractor

logger = logging.getLogger('asr.pipeline')

try:
    from warprnnt_tensorflow import rnnt_loss
except:
    logger.info("Could not import warp-rnnt loss")


class RNNTPipeline(CTCPipeline):
    """
    Pipeline modifies preprocessing step to feed labels as model inputs as well.
    """

    def preprocess(self,
                   batch: Tuple[List[np.ndarray], List[str]],
                   is_extracted: bool,
                   augmentation: augmentation.Augmentation):
        """ Preprocess batch data to format understandable to a model. """
        data, transcripts = batch
        if is_extracted:  # then just align features
            feature_lengths = [len(feature_seq) for feature_seq in data]
            features = FeaturesExtractor.align(data)
        else:
            features, feature_lengths = self._features_extractor(data, return_lengths=True)
        features = augmentation(features) if augmentation else features
        feature_lengths = np.array(feature_lengths)

        labels = self._alphabet.get_batch_labels(transcripts)
        label_lengths = np.array([len(decoded_text) for decoded_text in transcripts])
        return (features, labels, feature_lengths, label_lengths), labels

    def predict(self, batch_audio: List[np.ndarray], **kwargs) -> List[str]:
        """ Get ready features, and make a prediction. All additional model inputs
        except for the first one are replaced by zero tensors
        Implementation from https://github.com/zhanghaobaba/RNN-Transducer/blob/786fa75ff65c8ce859183d3c67aa408ff7fdef13/model.py#L33
        """
        assert len(batch_audio) == 1
        self.model.reset_states()

        # Extract submodels from big model
        pred_net = self.model.get_layer('pred_net')
        encoder = self.model.get_layer('encoder')
        joint_net = self.model.get_layer('joint')
        embedding = self.model.get_layer('embedding')

        input_features = self._features_extractor(batch_audio)
        res_seq = []
        # i will keep track of how many
        ys = []

        remove_me = []
        for t, x_t in enumerate(encoder.predict(input_features.astype(np.float32))[0]):
            # x_t has shape [num_features]
            while True:
                # make one step in prediction network. y_i has shape [num_features]
                if len(ys) < len(res_seq) + 1:
                    # Embed last predicted label. First embedding is just zeros not created from any label.
                    if len(res_seq) > 0:
                        embedded_label = embedding(np.array([
                            [res_seq[-1]]
                        ]))[0][0]
                    else:
                        embedded_label = np.zeros(pred_net.input.shape[-1])
                    ys.append(pred_net.predict(np.array([
                        [embedded_label]
                    ]))[0, 0])

                next_label_preds = joint_net(np.array([
                    np.concatenate([x_t, ys[-1]], axis=0)
                ]))
                remove_me.append(x_t)
                next_label = np.argmax(next_label_preds[0])

                print(f"Predicted label {next_label}")

                if next_label == self.alphabet.blank_token or (len(res_seq) > 0 and next_label == res_seq[-1]):
                    pass
                else:
                    res_seq.append(next_label)
                break
        remove_me = np.array(remove_me)
        print(remove_me.mean(axis=0), remove_me.std(axis=0))
        predictions = self._alphabet.get_batch_transcripts([res_seq])
        return predictions

    def get_loss(self) -> Callable:
        """ The CTC loss using rnnt_loss. """

        def rnnt_loss_wrapper(labels, outputs):
            logit_lengths = tf.math.floordiv(
                self.model.inputs[2][:, 0] + tf.math.floormod(self.model.inputs[2][:, 0], 2), 2)
            logit_lengths = keras.backend.print_tensor(logit_lengths, 'logit_lengths')

            label_lengths = keras.backend.print_tensor(self.model.inputs[3][:, 0], 'label_lengths')
            labels_ = keras.backend.print_tensor(labels, 'labels')
            outputs_ = keras.backend.print_tensor(outputs, 'logits')
            return rnnt_loss(outputs,
                             labels,
                             logit_lengths,
                             label_lengths,
                             # keras.backend.print_tensor(tf.shape(outputs_), 'dddd'),
                             blank_label=self._alphabet.blank_token)

        return rnnt_loss_wrapper
