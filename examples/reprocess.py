# standard libraries
import os
import glob
import argparse
# third party libraries
import tqdm
# project libraries
from speech.utils import convert, data_helpers


def reprocess(dataset_name:str):
    """
    Dataset name should be a key in the dataset_dir.
    TODO: convert dataset_dir into dataset class with attributes
    corpus_name, dataset_name, path, pattern. 
    """
    dataset_dir = {
    "train-clean-100":{
        "path": "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100",
        "pattern":"*/*/*.wav"},
    "train-clean-360":{
        "path": "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-360",
        "pattern":"*/*/*.wav"},
    "train-other-500":{
        "path": "/home/dzubke/awni_speech/data/LibriSpeech/train-other-500",
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
    libsp_sets = ["train-clean-100", "train-clean-360", "train-other-500"]
    for key, value in dataset_dir.items():
        if key == dataset_name:
            audio_pattern = os.path.join(value["path"], value["pattern"])
            audio_files = glob.glob(audio_pattern)
            print(f"Processing {key}...")
            for audio_file in tqdm.tqdm(audio_files):
                # this if-else section is gross, use dataset class to remedy
                if dataset_name in libsp_sets:
                    dataname_skip_file = "librispeech"
                else:
                    dataname_skip_file = dataset_name
                if data_helpers.skip_file(dataname_skip_file, audio_file):
                    print(f"skipping: {audio_file}")
                    continue
                convert.convert_2channels(audio_file)
            print(f"Finished processing {key}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(
            description="reprocesses the datasets mainly to convert everything to single channel audio.")
    parser.add_argument("--dataset-name", type=str,
        help="dataset name as specified by keys in dataset_dir in reprocess function")
    args = parser.parse_args()

    reprocess(args.dataset_name)
