# standard libraries
import os
import glob
# project libraries
import convert


def reprocess_all():
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

    for values in dataset_dir.values():
        audio_pattern = os.path.join(values["path"], values["pattern"])
        audio_files = glob.glob(audio_pattern)
        for audio_file in audio_files:
            convert.convert_2channels(audio_file)

if __name__=="__main__":
    reprocess()
