import numpy as np
import automatic_speech_recognition as asr


def test_standardize_batch():
    features = np.random.normal(loc=2, size=[5, 20, 10])
    lengths = (np.ones(shape=[5]) * 20).astype('int32')
    standardized_ft = asr.features.FeaturesExtractor.standardize_batch(
        features, lengths, how="per_feature")
    assert standardized_ft.shape == (5, 20, 10)
    assert np.all(np.isclose(standardized_ft.mean(axis=2), 0))
    assert np.all(np.isclose(standardized_ft.std(axis=2), 1, atol=1e-4))

    standardized_gl = asr.features.FeaturesExtractor.standardize_batch(
        features, lengths, how="all_features")
    assert standardized_gl.shape == (5, 20, 10)
    assert np.all(
        np.isclose(standardized_gl.reshape([5, 200]).mean(axis=1), 0))
    assert np.all(
        np.isclose(standardized_gl.reshape([5, 200]).std(axis=1), 1,
                   atol=1e-4))


def test_align():
    X = asr.features.FeaturesExtractor.align([np.ones([50, 10]),
                                              np.zeros([70, 10]),
                                              np.ones([60, 10])], default=9)
    assert X.shape == (3, 70, 10)
    assert np.all(X[2, 60:, :] == 9)
