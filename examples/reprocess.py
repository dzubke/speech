# standard libraries
import os
import glob
import argparse
from multiprocessing import Pool
# third party libraries
import tqdm
# project libraries
from speech.utils import convert, data_helpers
from speech import dataset_info


def reprocess_all():
    all_datasets = dataset_info.AllDatasets()
    num_datasets = len(all_datasets.dataset_list)
    process_list = list()
    for dataset in all_datasets.dataset_list:
        print(f"Collecting {dataset.dataset_name}...")
        process_list.append((dataset.get_audio_files(), dataset.corpus_name))
        print(f"Finished collecting {dataset.dataset_name}")
    
    with Pool(num_datasets) as p:
        p.starmap(convert_glob, process_list)

def reprocess_one(dataset_name:str):
    """
    Dataset names should be consistent with class names in speech/dataset_info.py
    """
    # ensuring the dataset_name is capitalized
    dataset_name = dataset_name.capitalize()
    # initializing the dataset object specified by dataset_name
    dataset = eval("dataset_info."+dataset_name+"Dataset")()
    print(f"Processing {dataset.dataset_name}...")
    convert_glob((dataset.get_audio_files(), dataset.corpus_name)
    print(f"Finished processing {dataset.dataset_name}")


def convert_glob(audio_files:list, corpus_name:str):
    """
    Takes in a glob list of audio file names and applies convert
    to those files. The corpus_name is used to determine which
    files should be skipped
    """
    for audio_file in tqdm.tqdm(audio_files):
        # this if-else section is gross, use dataset class to remedy
        if data_helpers.skip_file(corpus_name, audio_file):
            print(f"skipping: {audio_file}")
            continue
        convert.convert_2channels(audio_file)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
            description="reprocesses the datasets mainly to convert everything to single channel audio.")
    parser.add_argument("--dataset-name", type=str,
        help="dataset name. Options: Librispeech100, Librispeech360, Librispeech500, Commonvoice, Tedlium, Voxforge, Tatoeba")
    parser.add_argument("--all", action='store_true', default=False, help="will reprocess all datasets")
    args = parser.parse_args()
    
    if args.all:
        reprocess_all()
    else:
        reprocess_one(args.dataset_name)
