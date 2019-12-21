import numpy as np
import pytest
from unittest import mock
import automatic_speech_recognition as asr


@pytest.fixture
def dataset() -> asr.dataset.Dataset:
    reference = mock.Mock()
    reference.index = np.arange(100)

    generator = asr.dataset.Dataset(reference, batch_size=5)
    generator.get_batch = mock.Mock(return_value='test')
    return generator


def test_len(dataset):
    assert len(dataset) == 20


def test_getitem(dataset):
    assert all(dataset[i] == 'test' for i in dataset.indices)
