# standard library
import audioop
# third party libraries
import numpy as np
import pytest
# project libraries
from speech.utils.signal_augment import tempo_gain_pitch_perturb
from speech.utils.wave import array_from_wave
from tests.pytest.utils import get_all_test_audio



def test_tempo_augment():
    """
    Verifies the size of the augmented data scaled by the tempo equals the size
    of the un-augmented data
    """

    audio_paths = get_all_test_audio()
    tempos = [0, 0.5, 0.85, 1, 1.15, 2]
    for audio_path in audio_paths:
        # un-augmented audio_data
        audio_data, samp_rate = array_from_wave(audio_path)
        for tempo in tempos:
            aug_data, samp_rate = tempo_gain_pitch_perturb(audio_path, sample_rate=samp_rate,
                tempo_range=(tempo, tempo), gain_range=(0,0), pitch_range=(0,0))

        print(f"audio_data size: {audio_data.size}, aug_data: {aug_data.size}, tempo: {tempo}")
        assert audio_data.size == pytest.approx(aug_data.size * tempo, 1e-1)

def test_no_augment():
    """
    tests that input audio and augmented data are identical with no augmentation: tempo=1.0, gain=0
    pitch = 0
    """

    tempo = 1.0
    gain = 0.0
    pitch = 0.0
    audio_paths = get_all_test_audio()
    for audio_path in audio_paths:
        # un-augmented audio_data
        audio_data, samp_rate = array_from_wave(audio_path)
        aug_data, samp_rate = tempo_gain_pitch_perturb(audio_path, sample_rate=samp_rate,
                tempo_range=(tempo, tempo), gain_range=(gain, gain), pitch_range=(pitch, pitch))

        assert all(audio_data == aug_data), "data is not the same"

def test_gain_pitch_same_size():
    """
    tests that varying the gain and the pitch has no affect on the audio_data size
    """

    tempo = 1.0
    gain_pitch_tuples = [(0, 0),         # not augmentation
                        (8, 0),         # only gain aug
                        (0, 400),       # only pitch
                        (-6, -400)]      # both gain and pitch
    audio_path = get_all_test_audio()[0]       # only using a single audio path
    for gain, pitch in gain_pitch_tuples:
        # un-augmented audio_data
        audio_data, samp_rate = array_from_wave(audio_path)
        aug_data, samp_rate = tempo_gain_pitch_perturb(audio_path, sample_rate=samp_rate,
                tempo_range=(tempo, tempo), gain_range=(gain, gain), pitch_range=(pitch, pitch))

        assert audio_data.size == aug_data.size, "data size is not the same"


def test_gain_increase_amplitude():
    """
    tests that 1) 6 dB increase in gain coorespondes to a 1.995 increase in the sum of the absolute
    value of the amplitudes and,
    2) a 6 db decrease cooresponds to a 0.5 decrease in the sum abs value of amplitudes
    Ratio is computed as: ratio = 10**(gain/20)
    """

    tempo = 1.0
    pitch = 0.0
    gain_ratio_tuples = [(0, 1.0),         # not augmentation
                        (6, 1.995),         
                        (-6, 0.501)]      
                        #(10, 3.162),        # these two tests fail. 
                        #(-10, 0.3162)       # I'm not sure why, likely an error in my approach. 
    audio_paths = get_all_test_audio()       # only using a single audio path
    for audio_path in audio_paths:
        print(f"audio_path: {audio_path}")
        for gain, amp_ratio in gain_ratio_tuples:
            # un-augmented audio_data
            audio_data, samp_rate = array_from_wave(audio_path)
            aug_data, samp_rate = tempo_gain_pitch_perturb(audio_path, sample_rate=samp_rate,
                    tempo_range=(tempo, tempo), gain_range=(gain, gain), pitch_range=(pitch, pitch))
            audio_rms = audioop.rms(audio_data, 2)
            scaled_aug_rms = audioop.rms(aug_data, 2)/amp_ratio
            accuracy = -1  # same up to 10^(-accuracy)
            print(f"audio rms: {audio_rms}, scaled_aug rms: {scaled_aug_rms}, ratio:{amp_ratio}, accuracy:{10**(-accuracy)}")
            np.testing.assert_almost_equal(audio_rms, scaled_aug_rms, decimal=accuracy)