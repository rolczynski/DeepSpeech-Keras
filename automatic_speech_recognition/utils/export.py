import tensorflow as tf
from tensorflow import keras
from .weights_manip import load_weights


class KerasTfLiteExporter:
    """
    Wrapper class around TFLiteConverter. It's main reason is load model weights from chk.
    """
    def __init__(self, model: keras.Model, chk_path: str, skip_on_load_fail=False):
        load_weights(model, chk_path, skip_on_load_fail)
        self.__dict__['_model'] = model
        # Converting keras model with tf2.0 has a lot of unifxed bugs. Converter should be ran with tf2.1
        self.__dict__['_converter'] = tf.lite.TFLiteConverter.from_keras_model(model)

    def __getattr__(self, key):
        return getattr(self._converter, key)

    def __setattr__(self, key, value):
        setattr(self._converter, key, value)

    def export(self, model_path: str):
        tflite_model = self._converter.convert()
        with open(model_path, 'wb+') as f:
            f.write(tflite_model)
