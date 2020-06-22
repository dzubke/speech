
# third party libraries
import pytest
# project libraries
from speech.utils.signal_augment import speed_vol_perturb
from speech.utils.wave import array_from_wave


def test_speed_vol_main():
    check_speed()


def check_speed():
    """
    Verifies the speed change from the output of speed_vol_perturb
    """
    audio_files = [
        "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/19/198/19-198-0034.wav", 
        "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/19/198/19-198-0000.wav"
    ]
    tempos = [0, 0.5, 0.85, 1, 1.15, 2]
    for audio_file in audio_files:
        audio, samp_rate = array_from_wave(audio_file)
        for tempo in tempos:
            aug_data = speed_vol_perturb(path, sample_rate=samp_rate,
                tempo_range=(tempo, tempo), gain_range=(-6, 8))
            #assert audio.size == pytest.approx(aug_audio.size * tempo, 1e-1)