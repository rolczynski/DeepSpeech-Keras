import numpy as np
import tensorflow as tf
import os
import automatic_speech_recognition as asr

dataset = asr.dataset.Audio.from_csv('./data/LibriDev/data.csv', batch_size=5)
dev_dataset = asr.dataset.Audio.from_csv('./data/LibriDev/data.csv', batch_size=5)
alphabet = asr.text.Alphabet(lang='en')
features_extractor = asr.features.FilterBanks(
    features_num=160,
    winlen=0.02,
    winstep=0.01,
    winfunc=np.hanning
)
model = asr.model.get_deepspeech2(
    input_dim=160,
    output_dim=29,
    rnn_units=800,
    is_mixed_precision=False,
)
optimizer = tf.optimizers.Adam(
    lr=1e-2,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)
decoder = asr.decoder.GreedyDecoder()
pipeline = asr.pipeline.CTCPipeline(
    alphabet, features_extractor, model, optimizer, decoder
)

learning_rate_scheduler = asr.callback.LearningRateScheduler(
    schedule=lambda epoch, lr: lr / np.power(1.2, epoch)
)
callbacks = [learning_rate_scheduler]
pipeline.fit(dataset, dev_dataset, epochs=20, callbacks=callbacks, verbose=1)
pipeline.save('./checkpoint')

test_dataset = asr.dataset.Audio.from_csv('test.csv', batch_size=1)
wer, cer = asr.evaluate.calculate_error_rates(pipeline, test_dataset)
print(f'WER: {wer}   CER: {cer}')
