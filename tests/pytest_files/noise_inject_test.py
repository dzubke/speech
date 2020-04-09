# standard libraries
import os
import glob
# third party libraries
import pytest
# project libraries
from speech.utils import wave, spec_augment, noise_injector, data_helpers
from speech import dataset_info

def test_datasets():
    dataset_name = "Commonvoice"
    dataset_namme = dataset_name.capitalize()
    # initializing the dataset object specified by dataset_name
    dataset = eval("dataset_info."+dataset_name+"Dataset")()   
    audio_pattern = os.path.join(dataset.audio_dir, dataset.pattern)
    audio_files = glob.glob(audio_pattern)
    noise_path = "/home/dzubke/awni_speech/data/background_noise/388338__uminari__short-walk-thru-a-noisy-street-in-a-mexico-city.wav"
    for audio_file in audio_files:
        #check_all_noise(audio_file)
        if data_helpers.skip_file(dataset.corpus_name, audio_file):
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

