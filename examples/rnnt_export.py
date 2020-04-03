import numpy as np
import tensorflow as tf
import automatic_speech_recognition as asr
import logging
from automatic_speech_recognition.utils import load_weights, KerasTfLiteExporter

logging.basicConfig(level=logging.DEBUG)

alphabet = asr.text.Alphabet(lang='en')
model = asr.model.get_rnnt(
    input_dim=160,
    vocab_size_pred=alphabet.size,
    is_mixed_precision=False,
    convert_tflite=True
)

exporter = KerasTfLiteExporter(model, './checkpoint/model.tf', True)
exporter.experimental_new_converter = True
exporter.allow_custom_ops = True
exporter.export('./model.tflite')
