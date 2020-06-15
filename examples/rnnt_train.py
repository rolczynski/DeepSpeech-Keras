import numpy as np
import tensorflow as tf
import automatic_speech_recognition as asr
import logging
tf.audio
logging.basicConfig(level=logging.DEBUG)

dataset = asr.dataset.Audio.from_csv('test.csv', batch_size=1)
dev_dataset = asr.dataset.Audio.from_csv('test.csv', batch_size=1)
alphabet = asr.text.Alphabet(lang='en')
features_extractor = asr.features.FilterBanks(
    features_num=160,
    sample_rate=16000,
    winlen=0.02,
    winstep=0.01,
    window=np.hanning
)
model = asr.model.get_rnnt(
    input_dim=160,
    num_layers_encoder=2,
    num_layers_pred=2,
    embed_size_pred=20,
    batch_size=1,
    vocab_size_pred=alphabet.size,
    convert_tflite=False
)
optimizer = tf.optimizers.Adam(
    lr=1e-4
)

# learning_rate_scheduler = asr.callback.LearningRateScheduler(
#     schedule=lambda epoch, lr: lr / np.power(1.01, epoch)
# )
# callbacks = [learning_rate_scheduler]
decoder = asr.decoder.SimpleDecoder()
pipeline = asr.pipeline.RNNTPipeline(
    alphabet, features_extractor, model, optimizer, decoder
)
pipeline.fit(dataset, dev_dataset, epochs=800)
# pipeline.save('./checkpoint')

test_dataset = asr.dataset.Audio.from_csv('test.csv', batch_size=1)
wer, cer = asr.evaluate.calculate_error_rates(pipeline, test_dataset)
print(f'WER: {wer}   CER: {cer}')
