
# third-party libaries
import logging
import numpy as np
import pytest
# project libraries
from speech.utils.wave import array_from_wave
from speech.utils.pitch_perturb import apply_pitch_perturb


def test_no_change():
    """
    tests that the output is the same as the input when no pitch augmentation offerss
    """
    audio_path = "/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/tests/pytest/test_audio/Speak-out.wav"
    audio_data, samp_rate = array_from_wave(audio_path)
    lower_range=0
    upper_range=0
    augmented_data = apply_pitch_perturb(audio_data, samp_rate, lower_range=lower_range, upper_range=upper_range)

    
    print("sum audio_data", audio_data.sum())
    print("mean audio_data", audio_data.mean())
    print("sum augmented_data", augmented_data.sum())
    print("mean augmented_data", augmented_data.mean())

    # absolute tolerance (atol) of 32 is 0.05% of the maximum range (65,536)
    np.testing.assert_allclose(audio_data, augmented_data, rtol=1e-03, atol=32e-00)


def test_zero_input():
    """
    """
    audio_data = np.empty((2,)).astype('int16')
    samp_rate  = 1600
    lower_range = 4
    upper_range = 4 
    
    augmented_data = apply_pitch_perturb(audio_data, samp_rate, lower_range=lower_range, upper_range=upper_range)
    np.testing.assert_allclose(audio_data, augmented_data, rtol=1e-03, atol=32e-00)
    assert audio_data.sum() == augmented_data.sum() == 0, "Error in test_zero_input"
    

def test_min_input():
    """
    """
    audio_data = np.empty((0,)).astype('int16')
    samp_rate  = 1600
    lower_range = 4
    upper_range = 4 
    
    lengths_to_test = [0,1]

    with pytest.raises(AssertionError) as execinfo:
        augmented_data = apply_pitch_perturb(audio_data, samp_rate, lower_range=lower_range, upper_range=upper_range)
        

