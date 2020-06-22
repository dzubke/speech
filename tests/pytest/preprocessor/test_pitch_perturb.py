
# third-party libaries
import logging
import numpy as np
import pytest
# project libraries
from speech.utils.wave import array_from_wave
from speech.utils.signal_augment import apply_pitch_perturb


def test_no_change():
    """
    tests that the output is the same as the input when no pitch augmentation offerss
    """
    audio_path = "../test_audio/Speak-out.wav"
    audio_data, samp_rate = array_from_wave(audio_path)
    lower_level=0
    upper_level=0
    augmented_data = apply_pitch_perturb(audio_data, samp_rate, 
                                            pitch_range=(lower_level, upper_level))

    
    print("sum audio_data", audio_data.sum())
    print("mean audio_data", audio_data.mean())
    print("sum augmented_data", augmented_data.sum())
    print("mean augmented_data", augmented_data.mean())

    # absolute tolerance (atol) of 32 is 0.05% of the maximum range (65,536)
    np.testing.assert_allclose(audio_data, augmented_data, rtol=1e-03, atol=32e-00)


def test_zero_input():
    """
    tests on an empty array
    """
    audio_data = np.empty((2,)).astype('int16')
    samp_rate  = 1600
    lower_level = 4
    upper_level = 4 
    
    augmented_data = apply_pitch_perturb(audio_data, samp_rate, 
                                            pitch_range=(lower_level, upper_level))
    np.testing.assert_allclose(audio_data, augmented_data, rtol=1e-03, atol=32e-00)
    assert audio_data.sum() == augmented_data.sum() == 0, "Error in test_zero_input"
    

def test_min_input():
    """
    tests that an empty array with shape (0,) will raise an AssertionError
    """
    audio_data = np.empty((0,)).astype('int16')
    samp_rate  = 1600
    lower_level = 4
    upper_level = 4 
    
    lengths_to_test = [0,1]

    with pytest.raises(AssertionError) as execinfo:
        augmented_data = apply_pitch_perturb(audio_data, samp_rate, 
                                            pitch_range=(lower_level, upper_level))
        

