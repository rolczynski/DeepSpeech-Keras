import os
import numpy as np
import tensorflow as tf
import logging
from keras.callbacks import Callback, TensorBoard, EarlyStopping

from deepspeech_keras.utils import save

logger = logging.getLogger('deepspeech')


class ResultKeeper(Callback):
    """ Save evaluation result and log the processing results. """

    def __init__(self, file_path):
        super().__init__()
        self.results = []
        self.file_path = file_path

    def _set_up_new_batch(self, *_):
        """ Set up the new list for batch results."""
        self.batch = []

    on_epoch_begin = _set_up_new_batch

    def _save_batch_result(self, index, logs={}):
        """ Add next batch loss. """
        loss = logs.get('loss')
        self.batch.append(loss)
        logger.info(f'Batch ({index}): {loss:.2f}')

    on_batch_end = _save_batch_result

    def _save_epoch_results(self, epoch, logs={}):
        """ Collect all information about each epoch. """
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        self.results.append([epoch, loss, val_loss, self.batch])
        logger.info(f'Epoch ({epoch}): {loss}   {val_loss}')
        save(self.results, self.file_path)
        logger.info(f'Evaluation results saved in {self.file_path}')

    on_epoch_end = _save_epoch_results


class CustomModelCheckpoint(Callback):
    """ Save model architecture and weights for the single or multi-gpu model. """

    def __init__(self, template_model, log_dir):
        """ The template model shares the same weights, but it is not distributed
        along different devices (GPU's). It does matter for parallel models. """
        super().__init__()
        self.log_dir = log_dir
        self.best_result = np.inf
        self.best_weights_path = None
        self.template_model = template_model

    def _create_log_directory(self, _):
        """ Create the directory where the checkpoints are saved. """
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)

    on_train_begin = _create_log_directory

    def _save_model_weights(self, epoch, logs={}):
        """ Save model with weights of the single-gpu template model. """
        val_loss = logs.get('val_loss')
        name = f'weights.{epoch + 1:02d}-{val_loss:.2f}.hdf5'
        file_path = os.path.join(self.log_dir, name)
        self.template_model.save(file_path, overwrite=True)
        if val_loss < self.best_result:
            self.best_result = val_loss
            self.best_weights_path = file_path

    on_epoch_end = _save_model_weights

    def _set_best_weights_to_model(self, history):
        """ Set best weights to the model. Checkpoint callback save the best
        weights path. """
        self.template_model.load_weights(self.best_weights_path)

    on_train_end = _set_best_weights_to_model


class CustomTensorBoard(TensorBoard):
    """ This callback enable to save the batch logs. Write images and grads are
    disable. The generator is required and not supported with fit_generator. """

    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.processed_batches = 0

    def _save_batch_loss(self, _, logs={}):
        """ Add value to the tensorboard event """
        loss = logs.get('loss')
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = loss
        summary_value.tag = 'Loss (each batch)'
        self.writer.add_summary(summary, self.processed_batches)
        self.writer.flush()
        self.processed_batches += 1

    on_batch_end = _save_batch_loss


class CustomEarlyStopping(EarlyStopping):
    """ The callback stops training if the minimal target is not achieved. """

    def __init__(self, **kwargs):
        mini_targets = kwargs.pop('mini_targets')
        self._mini_targets = mini_targets
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        """ Finish training if the `monitor` value is too high. """
        super().on_epoch_end(epoch, logs)
        current = logs.get(self.monitor)
        if epoch in self._mini_targets and current > self._mini_targets[epoch]:
            self.model.stop_training = True
