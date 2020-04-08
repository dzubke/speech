# standard libraries
import os
import glob
# third party libraries
import pytest
# project libraries
from speech.utils import wave, spec_augment, noise_injector, data_helpers

def test_datasets():
    dataset_dir = {
        "train-clean-100":{
             "path": "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100",
             "pattern":"*/*/*.wav"},
        "train-clean-360":{
            "path": "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-360",
            "pattern":"*/*/*.wav"},
        "train-other-500":{
            "path": "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-500",
            "pattern":"*/*/*.wav"},
        "tedlium":{
            "path": "/home/dzubke/awni_speech/data/tedlium/TEDLIUM_release-3/data/converted/wav/",
            "pattern":"*.wav"},
        "voxforge":{
            "path": "/home/dzubke/awni_speech/data/voxforge/archive/",
            "pattern":"*/*/*.wv"},
        "tatoeba":{
            "path": "/home/dzubke/awni_speech/data/tatoeba/tatoeba_audio_eng/audio/",
            "pattern":"*/*.wv"}
    }
        
    dataset_name = "tatoeba"
    dataset = dataset_dir[dataset_name]
    audio_pattern = os.path.join(dataset["path"], dataset["pattern"])
    audio_files = glob.glob(audio_pattern)
    noise_path = "/home/dzubke/awni_speech/data/background_noise/388338__uminari__short-walk-thru-a-noisy-street-in-a-mexico-city.wav"
    for audio_file in audio_files:
        #check_all_noise(audio_file)
        if data_helpers.skip_file(dataset_name, audio_file):
            print(f"skipping: {audio_file}")
            continue            
        try:
            check_length(audio_file, noise_path)
        except AssertionError:
            raise AssertionError(f"error in audio: {audio_file} and noise: {noise_path}")
def check_all_noise(audio_path):
    noise_dir = "/home/dzubke/awni_speech/data/background_noise/"
    pattern = "*.wav"
    noise_pattern = os.path.join(noise_dir, pattern)
    noise_files = glob.glob(noise_pattern)
    for noise_file in noise_files:
        check_length(audio_path, noise_file)


def check_length(audio_path:str, noise_path:str):
    audio_data, samp_rate = wave.array_from_wave(audio_path)
    audio_noise = noise_injector.inject_noise_sample(audio_data, samp_rate, noise_path, 
                    noise_level=0.5, logger=None)

