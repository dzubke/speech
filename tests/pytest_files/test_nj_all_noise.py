# third party libraries
import pytest
# project libraries
from speech import dataset_info
import utils

def test_all_noise():
    noise_dataset = dataset_info.NoiseDataset()
    noise_files = noise_dataset.get_audio_files()
    audio_17s ="/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/19/198/19-198-0034.wav"
    audio_2s = "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/19/198/19-198-0000.wav"
    test_audio = [audio_2s, audio_17s]
    print(f"number of noise files testing: {len(noise_files)}")
    for audio_file in test_audio:
        for noise_file in noise_files:
            try:
                utils.check_length(audio_file, noise_file)
            except AssertionError:
                raise AssertionError(f"audio: {audio_file} and noise: {noise_file}")
            except FileNotFoundError:
                raise FileNotFoundError(f"audio: {audio_file} and noise: {noise_file}")


def test_noise_level():
    noise_files = [
        "/home/dzubke/awni_speech/data/background_noise/100263_43834-lq.wav",
        "/home/dzubke/awni_speech/data/background_noise/101281_1148115-lq.wav",
        "/home/dzubke/awni_speech/data/background_noise/102547_1163166-lq.wav",
        "/home/dzubke/awni_speech/data/background_noise/elaborate_thunder-Mike_Koenig-1877244752.wav",
        "/home/dzubke/awni_speech/data/background_noise/violet_noise_2.wav"
    ]
    audio_17s ="/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/19/198/19-198-0034.wav"
    audio_2s = "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/19/198/19-198-0000.wav"
    test_audio = [audio_2s, audio_17s]
    # making a list of noise_levels form 0 to 1.15 in increments of 0.5
    noise_levels = [x/100 for x in range(0,120, 5)]     

    print(f"number of noise files testing: {len(noise_files)}")
    for audio_file in test_audio:
        for noise_file in noise_files:
            for noise_level in noise_levels:
                try:
                    utils.check_length(audio_file, noise_file, noise_level=noise_level)
                except AssertionError:
                    raise AssertionError(f"audio:{audio_file}, noise:{noise_file}, noise_level:{noise_level}")
                except FileNotFoundError:
                    raise FileNotFoundError(f"audio:{audio_file}, noise:{noise_file}, noise_level:{noise_level}")
