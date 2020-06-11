# third party libraries
import pytest
import numpy as np
# project libraries
from speech.utils import data_helpers
from speech.utils.wave import array_from_wave
from speech.utils.signal_augment import inject_noise_sample
from speech import dataset_info
import utils

def test_all_noise_audio():
    all_datasets = dataset_info.AllDatasets()
    noise_dataset = dataset_info.NoiseDataset()
    # inputs to inject_noise
    noise_files = noise_dataset.get_audio_files()
    logger = None
    noise_levels = (0,0.5)
    # run through each dataset and test each file
    for dataset in all_datasets.dataset_list:
        audio_files = dataset.get_audio_files()
        for audio_file in audio_files:
            if data_helpers.skip_file(dataset.corpus_name, audio_file):
                print(f"skipping: {audio_file}")
                continue
            audio_data, samp_rate = wave.array_from_wave(audio_file)
            noise_file = np.random.choice(noise_files)
            noise_level = np.random.uniform(*noise_levels)
            try:
                inject_noise_sample(audio_data, samp_rate, noise_file, noise_level,logger=None)
            except AssertionError:
                raise AssertionError(f"audio: {audio_file}, noise: {noise_path}, noise_level: {noise_level}")
            except FileNotFoundError:
                raise FileNotFoundError(f"audio: {audio_file}, noise: {noise_path}, noise_level: {noise_level}")


