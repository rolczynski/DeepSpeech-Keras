import pytest
import automatic_speech_recognition as asr


@pytest.mark.slow
def test_load_deepspeech2():
    pipeline = asr.load('deepspeech2', lang='en')
    sample = asr.utils.read_audio('../sample-en.wav')
    transcript = pipeline.predict([sample])
    assert transcript == ['the streets were narrow and unpaved but very fairly clean']
