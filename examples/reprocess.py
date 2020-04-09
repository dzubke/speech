# standard libraries
import os
import glob
import argparse
# third party libraries
import tqdm
# project libraries
from speech.utils import convert, data_helpers
from speech import dataset_info


def reprocess(dataset_name:str):
    """
    Dataset name should be a key in the dataset_dir.
    """
    # initializing the dataset object specified by dataset_name
    dataset = eval("dataset_info."+dataset_name+"Dataset")()
    audio_pattern = os.path.join(dataset.audio_dir, dataset.pattern)
    audio_files = glob.glob(audio_pattern)
    print(f"Processing {dataset.dataset_name}...")
    for audio_file in tqdm.tqdm(audio_files):
        # this if-else section is gross, use dataset class to remedy
        if data_helpers.skip_file(dataset.corpus_name, audio_file):
            print(f"skipping: {audio_file}")
            continue
        convert.convert_2channels(audio_file)
    print(f"Finished processing {dataset.dataset_name}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(
            description="reprocesses the datasets mainly to convert everything to single channel audio.")
    parser.add_argument("--dataset-name", type=str,
        help="dataset name. Sample options: Libsp100, Commonvoice, Tedlium. See dataset_info.py for full options")
    args = parser.parse_args()

    reprocess(args.dataset_name)
