import automatic_speech_recognition as asr

train_gen = asr.generator.SourceGenerator.from_csv('train.csv', batch_size=32)
dev_gen = asr.generator.SourceGenerator.from_csv('dev.csv', batch_size=32)
polish_alphabet = asr.text.Alphabet(lang='pl')
filer_banks = asr.features.FilterBanks(
    samplerate=16000,
    winlen=0.025,
    winstep=0.01,
    nfilt=80,
    winfunc='hamming'
)
spec_augment = asr.augmentation.SpecAugment(
    F=40,
    mf=1,
    Tmin=10,
    Tmax=30,
    mt=5
)
deepspeech = asr.model.DeepSpeech(
    input_dim=80,
    output_dim=36,
    context=7,
    units=1024
)
adam_optimizer = asr.optimizer.Adam(
    lr=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=0.00000001
)
tf_decoder = asr.decoder.TensorflowDecoder(beam_size=1024)
pipeline = asr.pipeline.CTCPipeline(
    alphabet=polish_alphabet,
    model=deepspeech,
    optimizer=adam_optimizer,
    decoder=tf_decoder,
    features_extractor=filer_banks
)
pipeline.fit(train_gen, dev_gen, epochs=25, augmentation=spec_augment)
pipeline.save('/checkpoint')

eval_gen = asr.generator.SourceGenerator.from_csv('eval.csv')
wer, cer = asr.evaluate.calculate_error_rates(pipeline, eval_gen)
print(f'WER: {wer}   CER: {cer}')
