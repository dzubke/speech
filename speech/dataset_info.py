# stanard library
import os
import json
# project library
from speech.utils import data_helpers

class Dataset():
    def __init__(self):
        self.corpus_name = str()
        self.dataset_name = str()
        self.audio_dir = str()
        self.pattern = str()
        self.json = str()

    def get_audio_files(self)->list:
        """
        returns a list of the audio files in the json file
        """
        audio_files = list()
        with open(self.json) as fid:
            for line in fid:
                line = json.loads(line)
                audio_files.append(line.get("audio"))
        return audio_files

    def get_duration(self)->float:
        """
        returns a float of the total duration in hours of
        all the audio in the dataset
        """
        tot_duration = 0.0
        with open(self.json) as fid:
            for line in fid:
                line = json.loads(line)
                tot_duration += line.get("duration")/3600.0  # hours
        return tot_duration

    def files_from_pattern(self)->list:
        """
        returns a list of the audio files in the dataset based on the pattern attribute
        """
        return data_helpers.get_files(self.audio_dir, self.pattern)


class AllDatasets():
    def __init__(self):
        self.dataset_list = [
            Librispeech100Dataset(), Librispeech360Dataset(), 
            Librispeech500Dataset(),LibrispeechTestCleanDataset(), 
            LibrispeechTestOtherDataset(), LibrispeechDevCleanDataset(),
            LibrispeechDevOtherDataset(), TedliumDataset(), 
            CommonvoiceDataset(), VoxforgeDataset(), TatoebaDataset()
        ]
                            
class LibrispeechDataset(Dataset):
    def __init__(self):
        self.corpus_name = "librispeech"
        self.pattern = "*/*/*.wav"
        self.base_dir = "/home/dzubke/awni_speech/data/LibriSpeech/"

class Librispeech100Dataset(LibrispeechDataset):
    def __init__(self):
        super(Librispeech100Dataset, self).__init__()
        self.dataset_name = "train-clean-100"
        self.audio_dir = os.path.join(self.base_dir, self.dataset_name)
        self.json = "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100.json"

class Librispeech360Dataset(LibrispeechDataset):
    def __init__(self):
        super(Librispeech360Dataset, self).__init__()
        self.dataset_name = "train-clean-360"
        self.audio_dir = os.path.join(self.base_dir, self.dataset_name)
        self.json = "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-360.json"

class Librispeech500Dataset(LibrispeechDataset):
    def __init__(self):
        super(Librispeech500Dataset, self).__init__()
        self.dataset_name = "train-other-500"
        self.audio_dir = os.path.join(self.base_dir, self.dataset_name)
        self.json = "/home/dzubke/awni_speech/data/LibriSpeech/train-other-500.json"

class LibrispeechTestCleanDataset(LibrispeechDataset):
    def __init__(self):
        super(LibrispeechTestCleanDataset, self).__init__()
        self.dataset_name = "test-clean"
        self.audio_dir = os.path.join(self.base_dir, self.dataset_name)
        self.json = "/home/dzubke/awni_speech/data/LibriSpeech/test-clean.json"

class LibrispeechTestOtherDataset(LibrispeechDataset):
    def __init__(self):
        super(LibrispeechTestOtherDataset, self).__init__()
        self.dataset_name = "test-other"
        self.audio_dir = os.path.join(self.base_dir, self.dataset_name)
        self.json = "/home/dzubke/awni_speech/data/LibriSpeech/test-other.json"

class LibrispeechDevCleanDataset(LibrispeechDataset):
    def __init__(self):
        super(LibrispeechDevCleanDataset, self).__init__()
        self.dataset_name = "dev-clean"
        self.audio_dir = os.path.join(self.base_dir, self.dataset_name)
        self.json = "/home/dzubke/awni_speech/data/LibriSpeech/dev-clean.json"

class LibrispeechDevOtherDataset(LibrispeechDataset):
    def __init__(self):
        super(LibrispeechDevOtherDataset, self).__init__()
        self.dataset_name = "dev-other"
        self.audio_dir = os.path.join(self.base_dir, self.dataset_name)
        self.json = "/home/dzubke/awni_speech/data/LibriSpeech/dev-other.json"

class CommonvoiceDataset(Dataset):
    def __init__(self):
        self.corpus_name = "common-voice"
        self.dataset_name = "common-voice"
        self.audio_dir = "/home/dzubke/awni_speech/data/common-voice/clips/"
        self.pattern = "*.wv"
        self.json = "/home/dzubke/awni_speech/data/common-voice/validated-25-maxrepeat.json"

class TedliumDataset(Dataset):
    def __init__(self):
        self.corpus_name = "tedlium"
        self.dataset_name = "tedlium"
        self.audio_dir = "/home/dzubke/awni_speech/data/tedlium/TEDLIUM_release-3/data/converted/wav/"
        self.pattern = "*.wav"
        self.json = "/home/dzubke/awni_speech/data/tedlium/TEDLIUM_release-3/train.json"

class TedliumDevDataset(Dataset):
    def __init__(self):
        self.corpus_name = "tedlium"
        self.dataset_name = "tedlium-dev"
        self.audio_dir = "/home/dzubke/awni_speech/data/tedlium/TEDLIUM_release-3/legacy/dev/converted/wav/"
        self.pattern = "*.wav"
        self.json = "/home/dzubke/awni_speech/data/tedlium/TEDLIUM_release-3/dev.json"

class TedliumTestDataset(Dataset):
    def __init__(self):
        self.corpus_name = "tedlium"
        self.dataset_name = "tedlium-test"
        self.audio_dir = "/home/dzubke/awni_speech/data/tedlium/TEDLIUM_release-3/legacy/test/converted/wav/"
        self.pattern = "*.wav"
        self.json = "/home/dzubke/awni_speech/data/tedlium/TEDLIUM_release-3/test.json"

class VoxforgeDataset(Dataset):
    def __init__(self):
        self.corpus_name = "voxforge"
        self.dataset_name = "voxforge"
        self.audio_dir = "/home/dzubke/awni_speech/data/voxforge/archive/"
        self.pattern = "*/*/*.wv"
        self.json = "/home/dzubke/awni_speech/data/voxforge/all.json"

class TatoebaDataset(Dataset):
    def __init__(self):
        self.corpus_name = "tatoeba"
        self.dataset_name = "tatoeba"
        self.audio_dir = "/home/dzubke/awni_speech/data/tatoeba/tatoeba_audio_eng/audio/"
        self.pattern = "*/*.wv"
        self.json = "/home/dzubke/awni_speech/data/tatoeba/tatoeba_audio_eng/sentences_with_audio.json"

class NoiseDataset(Dataset):
    def __init__(self):
        self.corpus_name = "noise"
        self.dataset_name = "noise"
        self.audio_dir = "/home/dzubke/awni_speech/data/background_noise/"
        self.pattern = "*.wav"

class TestNoiseDataset(Dataset):
    def __init__(self):
        self.corpus_name = "noise"
        self.dataset_name = "test-noise"
        self.audio_dir = "/home/dzubke/awni_speech/data/background_noise/new_20200410/"
        self.pattern = "*.wav"
