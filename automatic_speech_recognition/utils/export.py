import tensorflow as tf
from tensorflow import keras
from .weights_manip import load_weights
import numpy as np
import logging

logger = logging.getLogger('asr.utils.export')


class KerasTfLiteExporter:
    """
    Wrapper class around TFLiteConverter. It's main reason is load model weights from chk.
    """

    def __init__(self, model: keras.Model, chk_path: str = None, skip_on_load_fail=False):
        if chk_path:
            load_weights(model, chk_path, skip_on_load_fail)
        self.__dict__['_model'] = model
        # Converting keras model with tf2.0 has a lot of unfixed bugs. Converter should be ran with tf2.1
        self.__dict__['_converter'] = tf.lite.TFLiteConverter.from_keras_model(model)

    def __getattr__(self, key):
        return getattr(self._converter, key)

    def __setattr__(self, key, value):
        setattr(self._converter, key, value)

    def export(self, model_path: str):
        tflite_model = self._converter.convert()
        with open(model_path, 'wb+') as f:
            f.write(tflite_model)
        self.check_tflite(model_path)

    def check_tflite(self, model_path: str):
        """
        Invokes tflite model and checks that outputs on random tensors are same with self.model
        :param model_path:
        :return:
        """
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])

        keras_output = self._model.predict(input_data)
        if np.allclose(tflite_output, keras_output, atol=1e-6):
            logger.error("Test successful. Tflite and Keras give similar results")
        else:
            logger.error("CAREFUL!!! TFLITE AND KERAS OUTPUTS ARE DIFFERENT")
            logger.error(f'MSE of outputs: {((keras_output - tflite_output) ** 2).mean()}')
