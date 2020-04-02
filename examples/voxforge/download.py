
# standard libraries
import argparse
import os
import tarfile
import urllib.request
# third party libraries
import tqdm


EXT = ".tar.gz"
DATA_URL = "https://s3.us-east-2.amazonaws.com/common-voice-data-download/voxforge_corpus_v1.0.0.tar.gz"
LEX_URL ="http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Lexicon/VoxForge.tgz"


def main(output_dir:str):
    save_dir = download_extract(output_dir)
    extract_samples(save_dir)

def download_extract(output_dir:str):
    save_dir = os.path.join(output_dir,"voxforge")
    download_dict = {"data": DATA_URL, "lexicon": LEX_URL}
    for name, url in download_dict.items():
        save_path = os.path.join(save_dir, name+EXT)
        print(f"Downloading: {name}...")
        urllib.request.urlretrieve(url, filename=save_path)
        print(f"Extracting: {name}...")
        with tarfile.open(save_path) as tf:
            tf.extractall(path=save_dir)
        os.remove(save_path)
        print(f"Processed: {name}")
    return save_dir

def extract_samples(save_dir:str):
    pattern = "*.tgz"
    sample_dir = os.path.join(save_dir,"archive")
    tar_path = os.path.join(sample_dir, pattern)
    tar_files = glob.glob(tar_path)
    print("Extracting and removing sample files...")
    for tar_file in tqdm(tar_files):
        with tarfile.open(tar_file) as tf:
            tf.extractall(path=sample_dir)
        os.remove(tar_file)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Download voxforge dataset.")

    parser.add_argument("--output_dir",
        help="The dataset is saved in <output_dir>/voxforge.")
    args = parser.parse_args()

    main(args.output_dir)