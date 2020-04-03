# standard libraries
import argparse
import os
import tarfile
import urllib.request
# third party libraries
import tqdm



class Downloader(object):

    def __init__(self, output_dir, dataset_name):
        self.output_dir = output_dir
        self.dataset_name = dataset_name.lower()
        self.download_dict = dict()

    def download_dataset(self):
        save_dir = self.download_extract()
        self.extract_samples(save_dir)

    def download_extract(self):
        """
        Standards method to download and extract zip file
        """
        save_dir = os.path.join(self.output_dir, self.dataset_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for name, url in self.download_dict.items():
            if name == "data":
                if os.path.exists(os.path.join(save_dir, self.data_dirname)):
                    print("Skipping data download")
                    continue
            save_path = os.path.join(save_dir, name + ".tar.gz")
            print(f"Downloading: {name}...")
            urllib.request.urlretrieve(url, filename=save_path)
            print(f"Extracting: {name}...")
            with tarfile.open(save_path) as tf:
                tf.extractall(path=save_dir)
            os.remove(save_path)
            print(f"Processed: {name}")
        return save_dir
    
    def extract_samples(self, save_dir:str):
        """
        Most datasets won't need to extract samples
        """
        pass
    

class VoxforgeDownloader(Downloader):

    def __init__(self, output_dir, dataset_name):
        super(VoxforgeDownloader, self).__init__(output_dir, dataset_name)
        self.download_dict = {
            "data": "https://s3.us-east-2.amazonaws.com/common-voice-data-download/voxforge_corpus_v1.0.0.tar.gz",
            "lexicon": "http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Lexicon/VoxForge.tgz"
        }
        self.data_dirname = "archive"


    def extract_samples(save_dir:str):
        """
        All samples are zipped in their own tar files.
        This function collects all the tar filenames and
        unzips themm.
        """
        pattern = "*.tgz"
        sample_dir = os.path.join(save_dir, self.data_dirname)
        tar_path = os.path.join(sample_dir, pattern)
        tar_files = glob.glob(tar_path)
        print("Extracting and removing sample files...")
        for tar_file in tqdm(tar_files):
            with tarfile.open(tar_file) as tf:
                tf.extractall(path=sample_dir)
            os.remove(tar_file)

class TatoebaDownloader(Downloader):

    def __init__(self, output_dir, dataset_name):
        super(TatoebaDownloader, self).__init__(output_dir, dataset_name)
        self.download_dict = { 
            "data": "https://downloads.tatoeba.org/audio/tatoeba_audio_eng.zip"
        }
        self.data_dirname = "audio"


class CommonvoiceDownloader(Downloader):

    def __init__(self, output_dir, dataset_name):
        super(CommonvoiceDownloader, self).__init__(output_dir, dataset_name)
        self.download_dict = {
            "data":"https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/en.tar.gz"
        }
        self.data_dirname = "clips"
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Download voxforge dataset.")

    parser.add_argument("--output-dir",
        help="The dataset is saved in <output-dir>/voxforge.")
    parser.add_argument("--dataset-name", type=str,
        help="Name of dataset with a capitalized first letter.")
    args = parser.parse_args()

    downloader = eval(args.dataset_name+"Downloader")(args.output_dir, args.dataset_name)
    print(f"type: {type(downloader)}")
    downloader.download_dataset()
