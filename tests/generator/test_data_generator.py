import os
import pytest
import pandas as pd
import automatic_speech_recognition as asr


@pytest.fixture
def generator() -> asr.generator.DataGenerator:
    directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(directory, '..', 'features', 'sample.wav')
    return asr.generator.DataGenerator(pd.DataFrame({
        'path': [file_path for i in range(10)],
        'transcript': ['abc' for i in range(10)],
    }), batch_size=3)


def test_get_batch(generator):
    assert len(generator) == 3
