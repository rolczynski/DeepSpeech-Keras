import os
import pytest
import automatic_speech_recognition as asr


@pytest.fixture
def alphabet() -> asr.text.Alphabet:
    return asr.text.Alphabet(lang='pl')


def test_contains(alphabet):
    assert 'a' in alphabet
    assert '9' not in alphabet


def test_label_from_string(alphabet):
    assert alphabet.label_from_string(' ') == 0
    assert alphabet.label_from_string('a') == 1
    assert alphabet.label_from_string('') == 35 == alphabet.blank_token


def test_string_from_label(alphabet):
    assert alphabet.string_from_label(alphabet.blank_token) == ''


def test_get_batch_labels(alphabet):
    transcripts = ['ala', 'ala kupiła', 'ala kupiła kota']
    batch_labels = alphabet.get_batch_labels(transcripts)
    assert batch_labels.shape == (3, len('ala kupiła kota'))


def test_get_batch_transcripts(alphabet):
    transcripts = ['ala', 'ala kupiła', 'ala kupiła kota']
    batch_labels = alphabet.get_batch_labels(transcripts)
    assert alphabet.get_batch_transcripts(batch_labels) == transcripts


def test_save_load(alphabet):
    file_path = 'alphabet.bin'
    asr.utils.save(alphabet, file_path)
    del alphabet
    alphabet = asr.utils.load(file_path)
    assert 'a' in alphabet
    assert '9' not in alphabet
    os.remove(file_path)
