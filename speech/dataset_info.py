# stanard library
import os
# project library
from speech.utils import data_helpers

class Dataset():
    def __init__(self, corpus_name:str, dataset_name:str, audio_dir:str, pattern:str):
        self.corpus_name = corpus_name
        self.dataset_name = dataset_name
        self.audio_dir = audio_dir
        self.pattern = pattern

    def get_audio_files(self):
        """
        returns a list of the audio files in the dataset based on the pattern attribute
        """
        return data_helpers.get_files(self.audio_dir, self.pattern)

class AllDatasets():
    def __init__(self):
        self.dataset_list = [Librispeech100Dataset(), Librispeech360Dataset(), Librispeech500Dataset(),
                            LibrispeechTestCleanDataset(), LibrispeechTestOtherDataset(), LibrispeechDevCleanDataset(),
                            LibrispeechDevOtherDataset(), TedliumDataset(), CommonvoiceDataset(), VoxforgeDataset(),
                            TatoebaDataset()]
                            
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

class Librispeech360Dataset(LibrispeechDataset):
    def __init__(self):
        super(Librispeech360Dataset, self).__init__()
        self.dataset_name = "train-clean-360"
        self.audio_dir = os.path.join(self.base_dir, self.dataset_name)

class Librispeech500Dataset(LibrispeechDataset):
    def __init__(self):
        super(Librispeech500Dataset, self).__init__()
        self.dataset_name = "train-other-500"
        self.audio_dir = os.path.join(self.base_dir, self.dataset_name)

class LibrispeechTestCleanDataset(LibrispeechDataset):
    def __init__(self):
        super(LibrispeechTestCleanDataset, self).__init__()
        self.dataset_name = "test-clean"
        self.audio_dir = os.path.join(self.base_dir, self.dataset_name)

class LibrispeechTestOtherDataset(LibrispeechDataset):
    def __init__(self):
        super(LibrispeechTestOtherDataset, self).__init__()
        self.dataset_name = "test-other"
        self.audio_dir = os.path.join(self.base_dir, self.dataset_name)

class LibrispeechDevCleanDataset(LibrispeechDataset):
    def __init__(self):
        super(LibrispeechDevCleanDataset, self).__init__()
        self.dataset_name = "dev-clean"
        self.audio_dir = os.path.join(self.base_dir, self.dataset_name)

class LibrispeechDevOtherDataset(LibrispeechDataset):
    def __init__(self):
        super(LibrispeechDevOtherDataset, self).__init__()
        self.dataset_name = "dev-other"
        self.audio_dir = os.path.join(self.base_dir, self.dataset_name)

class CommonvoiceDataset(Dataset):
    def __init__(self):
        self.corpus_name = "common-voice"
        self.dataset_name = "common-voice"
        self.audio_dir = "/home/dzubke/awni_speech/data/common-voice/clips/"
        self.pattern = "*.wv"

class TedliumDataset(Dataset):
    def __init__(self):
        self.corpus_name = "tedlium"
        self.dataset_name = "tedlium"
        self.audio_dir = "/home/dzubke/awni_speech/data/tedlium/TEDLIUM_release-3/data/converted/wav/"
        self.pattern = "*.wav"

class TedliumDevDataset(Dataset):
    def __init__(self):
        self.corpus_name = "tedlium"
        self.dataset_name = "tedlium-dev"
        self.audio_dir = "/home/dzubke/awni_speech/data/tedlium/TEDLIUM_release-3/legacy/dev/converted/wav/"
        self.pattern = "*.wav"

class TedliumTestDataset(Dataset):
    def __init__(self):
        self.corpus_name = "tedlium"
        self.dataset_name = "tedlium-test"
        self.audio_dir = "/home/dzubke/awni_speech/data/tedlium/TEDLIUM_release-3/legacy/test/converted/wav/"
        self.pattern = "*.wav"

class VoxforgeDataset(Dataset):
    def __init__(self):
        self.corpus_name = "voxforge"
        self.dataset_name = "voxforge"
        self.audio_dir = "/home/dzubke/awni_speech/data/voxforge/archive/"
        self.pattern = "*/*/*.wv"

class TatoebaDataset(Dataset):
    def __init__(self):
        self.corpus_name = "tatoeba"
        self.dataset_name = "tatoeba"
        self.audio_dir = "/home/dzubke/awni_speech/data/tatoeba/tatoeba_audio_eng/audio/"
        self.pattern = "*/*.wv"


class NoiseDataset(Dataset):
    def __init__(self):
        self.corpus_name = "noise"
        self.audio_dir = "/home/dzubke/awni_speech/data/background_noise"
        self.pattern = "*.wav"
