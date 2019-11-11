import numpy as np
import automatic_speech_recognition as asr


def test_align():
    X = asr.features.FeaturesExtractor.align([np.ones([50, 10]),
                                              np.zeros([70, 10]),
                                              np.ones([60, 10])], default=9)
    assert X.shape == (3, 70, 10)
    assert np.all(X[2, 60:, :] == 9)
