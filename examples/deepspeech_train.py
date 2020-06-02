import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import os
import automatic_speech_recognition as asr

dataset = asr.dataset.Audio.from_csv('./test.csv', batch_size=1, use_filesizes=False)
dev_dataset = asr.dataset.Audio.from_csv('./test.csv', batch_size=1, use_filesizes=False)
alphabet = asr.text.Alphabet(lang='en')
features_extractor = asr.features.MFCC(
    features_num=26,
    is_standardization=False,
    winlen=0.032,
    winstep=0.02,
)
model = asr.model.get_deepspeech(
    input_dim=26,
    output_dim=29,
    units=500,
    random_state=24,
    dropouts=(0, 0.0, 0.0, 0.0, 0),
    context=9,
    # rnn_units=800,
    # is_mixed_precision=False,
)
# model = asr.model.load_mozila_deepspeech('./data/myfrozen.pb')
optimizer = tf.optimizers.Adam(
    lr=5e-3,
    beta_1=0.9,
    beta_2=0.999
)
decoder = asr.decoder.GreedyDecoder()
pipeline = asr.pipeline.CTCPipeline(
    alphabet, features_extractor, model, optimizer, decoder
)
callbacks = []
pipeline.fit(dataset, dev_dataset, epochs=100, callbacks=callbacks)
pipeline.save('./checkpoint')

test_dataset = asr.dataset.Audio.from_csv('./test.csv', batch_size=1, use_filesizes=False)
wer, cer = asr.evaluate.calculate_error_rates(pipeline, test_dataset, print_pred=True)
print(f'WER: {wer}   CER: {cer}')
