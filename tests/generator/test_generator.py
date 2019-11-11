import os
from unittest.mock import Mock
import pytest
import pandas as pd
import automatic_speech_recognition as asr


@pytest.fixture
def generator() -> asr.generator.Generator:
    reference = Mock()
    reference.index.__len__ = Mock(return_value=101)
    generator = asr.generator.Generator(reference, batch_size=5)
    generator._get_batch = Mock(return_value='test')
    return generator


def test_generator(generator):
    assert len(generator) == 20
    assert generator.epoch == 0
    assert next(iter(generator)) == 'test'
    for i in range(10):
        list(generator)
    assert generator.epoch == 10
