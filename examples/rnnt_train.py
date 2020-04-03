import numpy as np
import tensorflow as tf
import automatic_speech_recognition as asr
import logging

logging.basicConfig(level=logging.DEBUG)

dataset = asr.dataset.Audio.from_csv('test.csv', batch_size=1)
dev_dataset = asr.dataset.Audio.from_csv('test.csv', batch_size=1)
alphabet = asr.text.Alphabet(lang='en')
features_extractor = asr.features.FilterBanks(
    features_num=160,
    winlen=0.02,
    winstep=0.01,
    winfunc=np.hanning
)
model = asr.model.get_rnnt(
    input_dim=160,
    vocab_size_pred=alphabet.size,
    is_mixed_precision=False,
    convert_tflite=True
)
optimizer = tf.optimizers.Adam(
    lr=1e-4,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)

decoder = asr.decoder.GreedyDecoder()
pipeline = asr.pipeline.RNNTPipeline(
    alphabet, features_extractor, model, optimizer, decoder
)
pipeline.fit(dataset, dev_dataset, epochs=1)
pipeline.save('./checkpoint')

test_dataset = asr.dataset.Audio.from_csv('test.csv', batch_size=1)
wer, cer = asr.evaluate.calculate_error_rates(pipeline, test_dataset)
print(f'WER: {wer}   CER: {cer}')
