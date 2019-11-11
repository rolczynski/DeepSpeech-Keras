import automatic_speech_recognition as asr

train_gen = asr.generator.DataGenerator.from_csv('train.csv', batch_size=32)
dev_gen = asr.generator.DataGenerator.from_csv('dev.csv', batch_size=32)
alphabet = asr.text.Alphabet(lang='pl')
features_extractor = asr.features.FilterBanks(
    samplerate=16000,
    winlen=0.025,
    winstep=0.01,
    nfilt=80,
    winfunc='hamming'
)
model = asr.model.DeepSpeech(
    input_dim=80,
    output_dim=36,
    context=7,
    units=1024
)
optimizer = asr.optimizer.Adam(
    lr=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=0.00000001
)
decoder = asr.decoder.TensorflowDecoder(beam_size=1024)
pipeline = asr.pipeline.CTCPipeline(
    alphabet, model, optimizer, decoder, features_extractor
)
pipeline.fit(train_gen, dev_gen, epochs=25)
pipeline.save('/checkpoint')

eval_gen = asr.generator.DataGenerator.from_csv('eval.csv')
wer, cer = asr.evaluate.calculate_error_rates(pipeline, eval_gen)
print(f'WER: {wer}   CER: {cer}')
