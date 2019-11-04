import automatic_speech_recognition as asr


# Case 1: Train a model, use either our model definition or build your own
train_gen = asr.generator.from_csv('train.csv')
dev_gen = asr.generator.from_files(['file.wav'])
alphabet = asr.text.Alphabet('pl')
features_extractor = asr.audio.fbank(
    winlen=0.025,
    winstep=0.01,
    nfilt=80,
    winfunc='hamming'
)
model = asr.models.CTCModel(name='DeepSpeech', size=1024)
optimizer = asr.optimizers.Adam(
    lr=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=0.00000001
)
decoder = asr.decoders.get_tensorflow_decoder(beam_size=1024)
pipeline = asr.pipeline.CTCPipeline(
    alphabet, features_extractor, model, optimizer, decoder
)
pipeline.fit(train_gen, dev_gen, epochs=25)
pipeline.save('checkpoint.pkl')

eval_gen = asr.generator.from_csv('eval.csv')
wer, cer = asr.evaluate(pipeline, eval_gen)
print(f'WER: {wer}   CER: {cer}')


# Case 2: There is already pre-trained model (polish or english)
pipeline = asr.pipeline.CTCPipeline.load('pl')


# More:
# using multi GPUs (a default is only one)
available_gpus = asr.get_available_gpus()
pipeline = asr.pipeline.CTCPipeline(
    alphabet, features_extractor, model, optimizer, decoder, gpus=available_gpus
)

# using pre-processed features
train_gen = asr.generator.from_prepared_features('features.hdf5')
...
pipeline = asr.pipeline.CTCPipeline(
    alphabet, model, optimizer, decoder, features_extractor=False
)

# closely monitor a training process (overwrite default callbacks)
callbacks = [
    asr.callbacks.TerminateOnNaN(),
    asr.callbacks.ResultKeeper(file_name='results.bin'),
    asr.callbacks.LearningRateScheduler(k=1.2, verbose=1)
]
pipeline = asr.pipeline.CTCPipeline(
    alphabet, features_extractor, model, optimizer, decoder, callbacks
)


# Contributing
#   change to tensorflow 2.0 (and tf.keras)
#   new architectures / entire pipelines: transducer, seq2seq (without/with attention)
#   new languages: if you can share your models or data, please contact us.
