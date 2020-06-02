import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import tensorflow as tf
import automatic_speech_recognition as asr
from automatic_speech_recognition.utils import load_weights, KerasTfLiteExporter

model = asr.model.load_mozilla_deepspeech('./data/myfrozen.pb', tflite_version=True)

exporter = KerasTfLiteExporter(model, skip_on_load_fail=True)
exporter.experimental_new_converter = True
exporter.allow_custom_ops = True
exporter.export('./model.tflite')
