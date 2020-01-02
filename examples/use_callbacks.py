import os
import numpy as np
import automatic_speech_recognition as asr

dataset = asr.dataset.Audio.from_csv('train.csv', batch_size=32)
dev_dataset = asr.dataset.Audio.from_csv('dev.csv', batch_size=32)
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
    is_mixed_precision=True
)
optimizer = asr.optimizer.Adam(
    lr=1e-4,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)
decoder = asr.decoder.GreedyDecoder()
pipeline = asr.pipeline.CTCPipeline(
    alphabet, features_extractor, model, optimizer, decoder
)
# Monitor training process
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)
learning_rate_scheduler = asr.callback.LearningRateScheduler(
    schedule=lambda epoch, lr: lr / np.power(1.2, epoch)
)
batch_logger = asr.callback.BatchLogger(
    file_path=os.path.join(model_dir, 'results.bin')
)
checkpoint = asr.callback.DistributedModelCheckpoint(
    template_model=pipeline.model,
    log_dir=os.path.join(model_dir, 'checkpoints')
)
callbacks = [learning_rate_scheduler, batch_logger, checkpoint]
pipeline.fit(dataset, dev_dataset, callbacks=callbacks, epochs=25)
pipeline.save(model_dir)

test_dataset = asr.dataset.Audio.from_csv('test.csv')
wer, cer = asr.evaluate.calculate_error_rates(pipeline, test_dataset)
print(f'WER: {wer}   CER: {cer}')
