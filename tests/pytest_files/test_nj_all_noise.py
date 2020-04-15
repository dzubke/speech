# third party libraries
import pytest
# project libraries
from speech import dataset_info
import testing_utils

def test_all_noise():
    noise_dataset = dataset_info.TestNoiseDataset()
    noise_files = noise_dataset.get_audio_files()
    audio_17s ="/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/19/198/19-198-0034.wav"
    audio_2s = "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/19/198/19-198-0000.wav"
    test_audio = [audio_2s, audio_17s]
    print(f"number of noise files testing: {len(noise_files)}")
    for audio_file in test_audio:
        for noise_file in noise_files:
            try:
                testing_utils.check_length(audio_file, noise_file)
            except AssertionError:
                raise AssertionError(f"audio: {audio_file} and noise: {noise_file}")
            except FileNotFoundError:
                raise FileNotFoundError(f"audio: {audio_file} and noise: {noise_file}")
