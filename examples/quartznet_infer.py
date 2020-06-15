import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import os
import automatic_speech_recognition as asr
import math

from automatic_speech_recognition.model.quartznet import load_nvidia_quartznet

alphabet = asr.text.Alphabet(lang='en')
features_extractor = asr.features.FilterBanks(
    features_num=64,
    sample_rate=16000,
    winlen=0.02,
    winstep=0.01,
    window="hann",
)
model = load_nvidia_quartznet(
    './data/JasperEncoder-STEP-247400.pt',
    './data/JasperDecoderForCTC-STEP-247400.pt')
optimizer = tf.optimizers.Adam(
    lr=1e-4,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)
decoder = asr.decoder.GreedyDecoder()
pipeline = asr.pipeline.CTCPipeline(
    alphabet, features_extractor, model, optimizer, decoder
)
audio, sample_rate = asr.utils.read_audio('../tests/sample-en.wav')
pipeline.predict([audio])
