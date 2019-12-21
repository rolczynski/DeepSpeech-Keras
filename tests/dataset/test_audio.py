import pytest
import pandas as pd
import automatic_speech_recognition as asr
from unittest import mock


@pytest.fixture
def dataset() -> asr.dataset.Audio:
    reference = pd.DataFrame({
        'path': [i for i in range(10)],
        'transcript': [f'transcript-{i}' for i in range(10)],
    })
    return asr.dataset.Audio(reference, batch_size=3)


def test_get_batch(dataset):
    module = 'automatic_speech_recognition.dataset.audio.utils'
    with mock.patch(module) as utils:
        utils.read_audio = lambda i: f'audio-{i}'
        batch_audio, transcripts = dataset.get_batch(index=1)
    a, b, c = transcripts
    assert b == 'transcript-4'
    a, b, c = batch_audio
    assert b == 'audio-4'
