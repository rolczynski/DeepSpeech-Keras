import numpy as np
import tensorflow as tf
import automatic_speech_recognition as asr
from automatic_speech_recognition.utils import load_weights, KerasTfLiteExporter

model = asr.model.get_deepspeech2(
    input_dim=160,
    output_dim=29,
    rnn_units=800,
    is_mixed_precision=False,
    convert_tflite=True
)

exporter = KerasTfLiteExporter(model, './checkpoint/model.tf', True)
exporter.experimental_new_converter = True
exporter.allow_custom_ops = True
exporter.export('./model.tflite')


