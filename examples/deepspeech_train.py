import numpy as np
import tensorflow as tf
import os
import automatic_speech_recognition as asr

alphabet = asr.text.Alphabet(lang='en')
features_extractor = asr.features.MFCC(
    features_num=26,
    sample_rate=16000,
    standardize=None,
    winlen=0.032,
    winstep=0.02,
)
model = asr.model.get_deepspeech(
    input_dim=26,
    output_dim=29,
    random_state=24,
    dropouts=(0, 0.0, 0.0, 0.0, 0),
    context=9,
    # rnn_units=800,
    # is_mixed_precision=False,
)
# model = asr.model.load_mozilla_deepspeech('./data/myfrozen.pb')
optimizer = tf.optimizers.Adam(
    lr=1e-3,
    beta_1=0.9,
    beta_2=0.999
)
decoder = asr.decoder.GreedyDecoder()
pipeline = asr.pipeline.CTCPipeline(
    alphabet, features_extractor, model, optimizer, decoder
)

callbacks = []
dataset = asr.dataset.Audio.from_csv('./data/dev-clean-index.csv', batch_size=10, use_filesizes=True)
dataset.sort_by_length()
dataset.shuffle_indices()
dev_dataset = asr.dataset.Audio.from_csv('./data/dev-clean-index.csv', batch_size=10, use_filesizes=True)
dev_dataset.sort_by_length()
dev_dataset.shuffle_indices()
pipeline.fit(dataset, dev_dataset, epochs=100, callbacks=callbacks)
# pipeline.save('./checkpoint')
# pipeline.save('./checkpoint')

test_dataset = asr.dataset.Audio.from_csv('./test.csv', batch_size=1, use_filesizes=False)
wer, cer = asr.evaluate.calculate_error_rates(pipeline, test_dataset, print_pred=True)
print(f'WER: {wer}   CER: {cer}')
