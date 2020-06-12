# standard libraries
import json
# third party libraries
import pytest
# project libraries
from speech.utils import data_helpers
from speech import dataset_info
import utils

def test_dataset():
    json_path = "/home/dzubke/awni_speech/data/common-voice/validated-25-maxrepeat.json"
    corpus_name = "common-voice"
    # initializing the dataset object specified by dataset_name
    noise_path = "/home/dzubke/awni_speech/data/background_noise/388338__uminari__short-walk-thru-a-noisy-street-in-a-mexico-city.wav"
    with open(json_path) as fid:
        for line in fid:
            sample=json.loads(line)
            audio_file = sample.get("audio")
            if data_helpers.skip_file(corpus_name, audio_file):
                print(f"skipping: {audio_file}")
                continue            
            try:
                utils.check_length(audio_file, noise_path)
            except:
                raise Exception(f"error in audio: {audio_file} and noise: {noise_path}")


