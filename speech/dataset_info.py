class Dataset():
    def __init__(self, corpus_name:str, dataset_name:str, audio_dir:str, pattern:str):
        self.corpus_name = corpus_name
        self.dataset_name = dataset_name
        self.audio_dir = audio_dir
        self.pattern = pattern

class LibrispeechDataset(Dataset):
    def __init__(self):
        self.corpus_name = "librispeech"
        self.pattern = "*/*/*.wav"        

class Libsp100Dataset(LibrispeechDataset):
    def __init__(self):
        super(Libsp100Dataset, self).__init__()
        self.dataset_name = "train-clean-100"
        self.audio_dir = "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/"

class Libsp360Dataset(LibrispeechDataset):
    def __init__(self):
        super(Libsp360Dataset, self).__init__()
        self.dataset_name = "train-clean-360"
        self.audio_dir = "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-360/"

class Libsp500Dataset(LibrispeechDataset):
    def __init__(self):
        super(Libsp500Dataset, self).__init__()
        self.dataset_name = "train-other-500"
        self.audio_dir = "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-500/"

class LibspTestCleanDataset(LibrispeechDataset):
    def __init__(self):
        super(LibspTestCleanDataset, self).__init__()
        self.dataset_name = "test-clean"
        self.audio_dir = "/home/dzubke/awni_speech/data/LibriSpeech/test-clean/"

class LibspTestOtherDataset(LibrispeechDataset):
    def __init__(self):
        super(LibspTestOtherDataset, self).__init__()
        self.dataset_name = "test-other"
        self.audio_dir = "/home/dzubke/awni_speech/data/LibriSpeech/test-other/"

class LibspDevCleanDataset(LibrispeechDataset):
    def __init__(self):
        super(LibspDevCleanDataset, self).__init__()
        self.dataset_name = "dev-clean"
        self.audio_dir = "/home/dzubke/awni_speech/data/LibriSpeech/dev-clean/"

class LibspDevOtherDataset(LibrispeechDataset):
    def __init__(self):
        super(LibspDevOtherDataset, self).__init__()
        self.dataset_name = "dev-other"
        self.audio_dir = "/home/dzubke/awni_speech/data/LibriSpeech/dev-other/"

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
