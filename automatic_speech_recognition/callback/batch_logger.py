import logging
from tensorflow import keras
from .. import utils
logger = logging.getLogger('asr.callback')


class BatchLogger(keras.callbacks.Callback):
    """ Log and save the processing batch results. """

    def __init__(self, file_path: str):
        super().__init__()
        self.batch = None
        self.results = []
        self.file_path = file_path

    def on_epoch_begin(self, logs=None):
        """ Set up the new list for batch results."""
        self.batch = []

    def on_train_batch_end(self, index: int, logs=None):
        """ Add next batch loss. """
        if not logs:
            return
        loss = logs.get('loss')
        self.batch.append(loss)
        logger.info(f'Batch ({index}): {loss:.2f}')

    def on_epoch_end(self, epoch: int, logs=None):
        """ Collect all information about each epoch. """
        if not logs:
            return
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        self.results.append([epoch, loss, val_loss, self.batch])
        logger.info(f'Epoch ({epoch}): {loss}   {val_loss}')
        utils.save(self.results, self.file_path)
