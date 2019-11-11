import os
import logging
import numpy as np
from tensorflow import keras
logger = logging.getLogger('asr.callback')


class DistributedModelCheckpoint(keras.callbacks.Callback):
    """ Save model architecture and weights for either the single or multi-gpu model. """

    def __init__(self, template_model: keras.Model, log_dir: str):
        """ The template model shares the same weights, but it is not distributed
        along different devices (GPUs). It does matter for parallel models. """
        super().__init__()
        self.log_dir = log_dir
        self.best_result = np.inf
        self.best_weights_path = None
        self.template_model = template_model

    def on_train_begin(self, logs=None):
        """ Create the directory where the checkpoints are saved. """
        if not os.path.isdir(self.log_dir):
            logger.info(f'Created the checkpoint directory: {os.path.abspath(self.log_dir)}')
            os.makedirs(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        """ Save model with weights of the single-gpu template model. """
        if not logs:
            return
        val_loss = logs.get('val_loss')
        name = f'weights.{epoch + 1:02d}-{val_loss:.2f}.h5'
        file_path = os.path.join(self.log_dir, name)
        self.template_model.save(file_path, overwrite=True)
        if val_loss < self.best_result:
            logger.info(f'Achieve the new best result: {val_loss:.2f}')
            self.best_result = val_loss
            self.best_weights_path = file_path

    def on_train_end(self, logs=None):
        """ Set best weights to the model. Checkpoint callback save the best
        weights path. """
        logger.info(f'Load the best weights: {self.best_weights_path}')
        self.template_model.load_weights(self.best_weights_path)
