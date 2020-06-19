# third party libraries
import pytest
# project libraries
from speech.utils import data_helpers
from speech import dataset_info
import utils

def test_dataset():
    dataset_name = "Commonvoice"
    # initializing the dataset object specified by dataset_name
    dataset = eval("dataset_info."+dataset_name+"Dataset")()   
    audio_files = dataset.get_audio_files()
    noise_path = "/home/dzubke/awni_speech/data/background_noise/388338__uminari__short-walk-thru-a-noisy-street-in-a-mexico-city.wav"
    for audio_file in audio_files:
        if data_helpers.skip_file(dataset.corpus_name, audio_file):
            print(f"skipping: {audio_file}")
            continue            
        try:
            utils.check_length(audio_file, noise_path)
        except AssertionError:
            raise AssertionError(f"error in audio: {audio_file} and noise: {noise_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"audio: {audio_file} and noise: {noise_file}")
        except:
            raise Exception(f"audio: {audio_file}, noise: {noise_path}")
