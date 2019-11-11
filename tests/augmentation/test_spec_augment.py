import numpy as np
import automatic_speech_recognition as asr
np.random.seed(1)


def test_call():
    features = np.random.random([3, 100, 40])
    spec_augment = asr.augmentation.SpecAugment(
        F=40,
        mf=1,
        Tmin=10,
        Tmax=30,
        mt=5
    )
    augmented = spec_augment(features)
    assert features.shape == augmented.shape
    assert not np.array_equal(features, augmented)


def test_mask_frequencies():
    features = np.random.random([100, 40])
    time, channels = features.shape
    means = features.mean(axis=1)
    augmented = asr.augmentation.SpecAugment.mask_frequencies(features.copy(), means, channels, F=20, mf=1)
    assert features.shape == augmented.shape

    is_masked = lambda x: [np.all(x[:, i] == means[i]) for i in range(channels)]
    assert sum(is_masked(augmented)) <= 50

    few_augmented = [asr.augmentation.SpecAugment.mask_frequencies(features.copy(), means, channels, F=20, mf=1)
                     for i in range(100)]
    checks = np.array([is_masked(augmented) for augmented in few_augmented])
    distribution = checks.sum(axis=0)
    counts, bins = np.histogram(distribution, bins=[0, 10, 20, 30, 40, 50, 60])
    assert np.array_equal(counts, [12, 6, 6, 9, 7, 0])


def test_mask_time():
    features = np.random.random([100, 40])
    time, channels = features.shape
    means = features.mean(axis=0)
    kwargs = dict(time=time, T_range=[10, 20], mt=3)
    augmented = asr.augmentation.SpecAugment.mask_time(features.copy(), means, **kwargs)
    assert features.shape == augmented.shape

    is_masked = lambda x: [np.all(x[i] == means) for i in range(time)]
    mask = is_masked(augmented)
    assert sum(mask) == 37

    few_augmented = [asr.augmentation.SpecAugment.mask_time(features.copy(), means, **kwargs)
                     for i in range(100)]
    masks = np.array([is_masked(augmented) for augmented in few_augmented])
    distribution = masks.sum(axis=0)
    counts, bins = np.histogram(distribution, bins=[0, 10, 20, 30, 40, 50, 60])
    assert np.array_equal(counts, [8, 6, 9, 20, 47, 10])
