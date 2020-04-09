class Dataset():
    def __init__(self, corpus_name:str, dataset_name:str, audio_dir:str, pattern:str):
        self.corpus_name = corpus_name
        self.dataset_name = dataset_name
        self.audio_dir = audio_dir
        self.pattern = pattern


class Librispeech100Dataset(Dataset):
    def __init__(self):
        self.corpus_name = "librispeech"
        self.dataset_name = "train-clean-100"
        self.audio_dir = "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/"
        self.pattern = "*/*/*.wav"


class Librispeech360Dataset(Dataset):
    def __init__(self):
        self.corpus_name = "librispeech"
        self.dataset_name = "train-clean-360"
        self.audio_dir = "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-360/"
        self.pattern = "*/*/*.wav"

class Librispeech500Dataset(Dataset):
    def __init__(self):
        self.corpus_name = "librispeech"
        self.dataset_name = "train-other-500"
        self.audio_dir = "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-500/"
        self.pattern = "*/*/*.wav"

class CommonvoiceDataset(Dataset):
    def __init__(self):
        self.corpus_name = "common-voice"
        self.dataset_name = "common-voice"
        self.audio_dir = "/home/dzubke/awni_speech/data/common-voice/clips/"
        self.pattern = "*.wav"

class TedliumDataset(Dataset):
    def __init__(self):
        self.corpus_name = "tedlium"
        self.dataset_name = "tedlium"
        self.audio_dir = "/home/dzubke/awni_speech/data/tedlium/TEDLIUM_release-3/data/converted/wav/"
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