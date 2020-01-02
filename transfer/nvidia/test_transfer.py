import numpy as np
import pytest
import transfer
import utils
import automatic_speech_recognition as asr


@pytest.fixture
def pipeline() -> asr.pipeline.CTCPipeline:
    transfer.transfer_open_seq2seq_deepspeech2(
        weights_file_name='nvidia-weights.bin', store_file_name='weights.h5'
    )
    deepspeech2 = asr.model.get_deepspeech2(input_dim=160, output_dim=29)
    deepspeech2.load_weights('weights.h5')
    alphabet_en = asr.text.Alphabet(lang='en')
    # padding audio signal is only required to compare activations
    spectrogram = asr.features.Spectrogram(
        features_num=160,
        samplerate=16000,
        winlen=0.02,
        winstep=0.01,
        winfunc=np.hanning,
        pad_audio_to=8
    )
    greedy_decoder = asr.decoder.GreedyDecoder()
    pipeline = asr.pipeline.CTCPipeline(
        alphabet=alphabet_en,
        model=deepspeech2,
        optimizer=None,  # Inference mode
        decoder=greedy_decoder,
        features_extractor=spectrogram
    )
    return pipeline


def correct(a, ref, atol) -> float:
    correct_weights = np.isclose(a, ref, atol=atol)
    correct = correct_weights.sum() / ref.size
    return correct


def test_transfer_open_seq2seq_deepspeech2(pipeline):
    nvidia = asr.utils.load('nvidia-activations.bin')
    model = pipeline.model

    # Test Feature Extraction process
    x = asr.utils.read_audio('../../tests/sample-en.wav')
    x = pipeline.features_extractor([x])
    assert x.shape == nvidia['input'].shape
    assert np.allclose(x, nvidia['input'], atol=1e-5)

    # Test Convolutional layers
    x = model.get_layer('lambda')(x)
    x = model.get_layer('conv_1')(x)
    # The flag in the batch normalization (conv_{i}_bn) is not required.
    # By default, use the trained moving mean and variance.
    x = model.get_layer('conv_1_bn')(x, training=False)
    x = model.get_layer('conv_1_relu')(x)
    assert x.shape == nvidia['conv_1_relu'].shape
    assert correct(x, nvidia['conv_1_relu'], atol=1e-4)
    x = model.get_layer('conv_2')(x)
    x = model.get_layer('conv_2_bn')(x, training=False)
    x = model.get_layer('conv_2_relu')(x)
    assert x.shape == nvidia['conv_2_relu'].shape
    assert correct(x, nvidia['conv_2_relu'], atol=1e-4)
    rnn_input = model.get_layer('reshape')(x)
    del x

    # Test Recurrent layers
    # This is a sanity check to understand CuDNN layer properties
    source_weights = asr.utils.load('nvidia-weights.bin')
    w = transfer.change_names(source_weights)
    hidden_states = {}
    x = rnn_input[0, ...].numpy()  # The single element in a batch
    for i in [1, 2, 3, 4, 5]:
        w_forward = w[f'bidirectional_{i}/forward']
        h_fw = np.stack(utils.numpy_gru(x, w_forward, units=800))

        w_backward = w[f'bidirectional_{i}/backward']
        x = np.flip(x, axis=0)
        h_bw = np.stack(utils.numpy_gru(x, w_backward, units=800))
        h_bw = np.flip(h_bw, axis=0)

        h = np.concatenate([h_fw, h_bw], axis=1)
        hidden_states[i] = h
        # Last hidden states are inputs to the next layer
        x = h
    x = np.expand_dims(x, axis=0)   # Add the batch axis
    assert x.shape == nvidia['bidirectional_5'].shape
    # The tolerance is reduced, because error is accumulated through five layers
    assert correct(x, nvidia['bidirectional_5'], atol=1e-2) > 0.999
    del x

    # Here we check if model has correct weights. The modified gru
    # function calculates activations similar as the GRU layer in our model.
    w = model.get_layer('bidirectional_1').get_weights()
    w_forward = {'kernel': w[0], 'recurrent_kernel': w[1], 'bias': w[2]}
    w_backward = {'kernel': w[3], 'recurrent_kernel': w[4], 'bias': w[5]}
    # Once again load the CNN output
    x = rnn_input[0, ...].numpy()  # The single element in a batch
    h_fw = np.stack(utils.numpy_gru_modified(x, w_forward, units=800))
    x = np.flip(x, axis=0)
    h_bw = np.stack(utils.numpy_gru_modified(x, w_backward, units=800))
    h_bw = np.flip(h_bw, axis=0)
    h = np.concatenate([h_fw, h_bw], axis=1)
    # We compare hidden states to our sanity check
    assert h.shape == hidden_states[1].shape
    assert correct(h, hidden_states[1], atol=1e-3) > 0.999
    del x

    # Final test of recurrent layers
    x = rnn_input.numpy()
    x = model.get_layer('bidirectional_1')(x)
    assert correct(x, hidden_states[1], atol=1e-3) > 0.999
    x = model.get_layer('bidirectional_2')(x)
    x = model.get_layer('bidirectional_3')(x)
    x = model.get_layer('bidirectional_4')(x)
    x = model.get_layer('bidirectional_5')(x)
    assert x.shape == nvidia['bidirectional_5'].shape
    assert correct(x, nvidia['bidirectional_5'], atol=1e-2) > 0.999

    # Test Encoder Layers
    x = model.get_layer('dense_1')(x)
    x = model.get_layer('dense_1_relu')(x)
    assert x.shape == nvidia['dense_1'].shape
    assert correct(x, nvidia['dense_1'], atol=1e-3) > 0.98
    x = model.get_layer('dense_2')(x)
    assert x.shape == nvidia['dense_2'].shape
    assert correct(x, nvidia['dense_2'], atol=1e-2) > 0.97

    sample = asr.utils.read_audio('../../tests/sample-en.wav')
    transcript = pipeline.predict([sample])
    assert transcript == ['the streets were narrow and unpaved but very fairly clean']
