# standard library
import pickle
# third-party libraries
import numpy as np
import pytest
from _pytest.fixtures import SubRequest
# project libraries
from tests.pytest.utils import get_all_test_audio
from speech.utils.signal_augment import synthetic_gaussian_noise_inject
from speech.utils.wave import array_from_wave


def test_high_snr_value():
    test_audio_paths = get_all_test_audio()
    snr_level=100
    # absolute tolerance is 1e-5 of the range of values in pcm16 format (2**16)
    atol = 2**16 * 1e-5
    for audio_path in test_audio_paths:
        audio_data, sr = array_from_wave(audio_path)
        augmented_data = synthetic_gaussian_noise_inject(audio_data, snr_range=(snr_level, snr_level))
        np.testing.assert_allclose(audio_data, augmented_data, rtol=1e-03, atol=atol)

def test_datatype():
    test_audio_paths = get_all_test_audio()
    snr_level = 30
    for audio_path in test_audio_paths:
        audio_data, sr = array_from_wave(audio_path)
        augmented_data = synthetic_gaussian_noise_inject(audio_data, snr_range=(snr_level, snr_level))
        assert augmented_data.dtype == "int16"

def test_zero_input():
    empty_data = np.empty((0,), dtype="int16")
    snr_level = 30
    synthetic_gaussian_noise_inject(empty_data, snr_range=(snr_level, snr_level))

def test_float_input_failure():
    float_data = np.arange(100, dtype=np.float64)
    snr_level=30
    with pytest.raises(AssertionError) as execinfo:
        synthetic_gaussian_noise_inject(float_data, snr_range=(snr_level, snr_level))


def test_regression_equal_pickle():
    """
    The pickle data is output from using the Speak-out.wav file with an snr_level = 30 and a random seed of zero
    """
    pickle_path = "../test_pickle/sythentic-gaussian-noise-inject_Speak-out_snr-30.pickle"
    with open(pickle_path, 'rb') as fid:
        pickle_data = pickle.load(fid)
    
    audio_path = "../test_audio/Speak-out.wav"
    snr_level = 30
    audio_data, sr = array_from_wave(audio_path)

    np.random.seed(0)
    augmented_data = synthetic_gaussian_noise_inject(audio_data, snr_range=(snr_level, snr_level))

    assert (augmented_data==pickle_data).sum() == augmented_data.size, "regression test fails"
