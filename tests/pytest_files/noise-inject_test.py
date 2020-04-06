import pytest
from speech.utils import wave, spec_augment
from speech.utils import noise_injector
import os
import glob

def test_file1_method1():
	x=5
	y=6
	assert x+1 == y,"test failed"
	assert x == y,"test failed"


def test_dataset():
	audio_dir = "/Users/dustin/CS/consulting/firstlayerai/data/LibriSpeech/train-clean-100/"
	pattern = "*/*/*.wav"
	audio_pattern = os.path.join(audio_dir, pattern)
	audio_files = glob.glob(audio_pattern)
	for audio_file in audio_files:
		check_all_noise(audio_file)

def check_all_noise(audio_path):
	noise_dir = "/Users/dustin/CS/consulting/firstlayerai/data/background_noise/new_audio/resampled/"
	pattern = "*.wav"
	noise_pattern = os.path.join(noise_dir, pattern)
	noise_files = glob.glob(noise_pattern)
	for noise_file in noise_files:
		check_length(audio_path, noise_file)


def check_length(audio_path:str, noise_path:str):
	audio_data, samp_rate = wave.array_from_wave(audio_path)
	audio_noise = noise_injector.inject_noise_sample(audio_data, samp_rate, noise_path, noise_level=0.5)


# def test_single_sample():
# 	audio_path = ""
# 	noise_path = ""
# 	check_length(audio_path, noise_path)