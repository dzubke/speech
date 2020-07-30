# standard libraries
import argparse
import glob
import os
from shutil import copyfile
import tarfile
import urllib.request
from zipfile import ZipFile
# third party libraries
import tqdm
# project libraries
from speech.utils.convert import to_wave



class Downloader(object):

    def __init__(self, output_dir, dataset_name):
        self.output_dir = output_dir
        self.dataset_name = dataset_name.lower()
        self.download_dict = dict()
        self.ext = ".tar.gz"


    def download_dataset(self):
        save_dir = self.download_extract()
        print(f"Data saved and extracted to: {save_dir}")
        self.extract_samples(save_dir)

    def download_extract(self):
        """
        Standards method to download and extract zip file
        """
        save_dir = os.path.join(self.output_dir, self.dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        for name, url in self.download_dict.items():
            if name == "data":
                if os.path.exists(os.path.join(save_dir, self.data_dirname)):
                    print("Skipping data download")
                    continue
    
            save_path = os.path.join(save_dir, name + self.ext)
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

        # downloads
        # https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2
        # https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences_CC0.tsv.bz2
        # https://downloads.tatoeba.org/exports/sentences_with_audio.tar.bz2
        # https://downloads.tatoeba.org/exports/user_languages.tar.bz2
        # https://downloads.tatoeba.org/exports/users_sentences.csv


    def download_extract(self):
        """
        Standards method to download and extract zip file
        """
        save_dir = os.path.join(self.output_dir,"commmon-voice")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for name, url in download_dict.items():
            if name == "data":
                if os.path.exists(os.path.join(save_dir, self.data_dirname)):
                    print("Skipping data download")
                    continue
            save_path = os.path.join(save_dir, name + ".tar.gz")
            print(f"Downloading: {name}...")
            urllib.request.urlretrieve(url, filename=save_path)
            print(f"Extracting: {name}...")
            raise NotImplementedError("Zip can't yet be extracted with tarfile code")
            with tarfile.open(save_path) as tf:
                tf.extractall(path=save_dir)
            os.remove(save_path)
            print(f"Processed: {name}")
        return save_dir


class TatoebaV2Downloader(Downloader):

    def __init__(self, output_dir, dataset_name):
        super(TatoebaDownloader, self).__init__(output_dir, dataset_name)
        self.download_dict = { 
            "data": "",
            "data_csv": "https://downloads.tatoeba.org/exports/sentences_with_audio.tar.bz2"
        }
        self.data_dirname = "audio"


    def download_extract(self):
        """
        Standards method to download and extract zip file
        """
        save_dir = os.path.join(self.output_dir,"commmon-voice")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for name, url in download_dict.items():
            if name == "data":
                if os.path.exists(os.path.join(save_dir, self.data_dirname)):
                    print("Skipping data download")
                    continue
            save_path = os.path.join(save_dir, name + ".tar.gz")
            print(f"Downloading: {name}...")
            urllib.request.urlretrieve(url, filename=save_path)
            print(f"Extracting: {name}...")
            raise NotImplementedError("Zip can't yet be extracted with tarfile code")
            with tarfile.open(save_path) as tf:
                tf.extractall(path=save_dir)
            os.remove(save_path)
            print(f"Processed: {name}")
        return save_dir




class CommonvoiceDownloader(Downloader):

    def __init__(self, output_dir, dataset_name):
        """
        A previous version of common voice (v4) can be downloaded here:
        "data":"https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/en.tar.gz"
        """
        super(CommonvoiceDownloader, self).__init__(output_dir, dataset_name)
        self.download_dict = {
            "data":"https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/en.tar.gz"
        }
        self.data_dirname = "clips"
    


class WikipediaDownloader(Downloader):

    def __init__(self, output_dir, dataset_name):
        """
        A previous version of common voice (v4) can be downloaded here:
        "data":"https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/en.tar.gz"
        """
        super(WikipediaDownloader, self).__init__(output_dir, dataset_name)
        self.download_dict = {
            "data":"https://www2.informatik.uni-hamburg.de/nats/pub/SWC/SWC_English.tar"
        }
        self.data_dirname = "not-used"
        self.ext = ".tar"
    


class Chime1Downloader(Downloader):

    def __init__(self, output_dir, dataset_name):
        super(Chime1Downloader, self).__init__(output_dir, dataset_name)
        # original dataset is in 2-channel. 
        # in this case, I have downloaded and transfered the data to the VM myself. 



class DemandDownloader(Downloader):

    def __init__(self, output_dir, dataset_name):
        """
        Limitations: 
            - feed_model_dir, the directory where the noise is fed to the model, is hard-coded
        """
        super(DemandDownloader, self).__init__(output_dir, dataset_name)
        self.download_dict = {}
        self.feed_model_dir = "/home/dzubke/awni_speech/data/noise/feed_to_model"
        self.load_download_dict()

    def load_download_dict(self):
        """
        Loads the download dictionary with download links and dataset names

        Assumptions: 
            Download links will still work in the future
        """

        noise_filenames = ["TMETRO_16k.zip", "TCAR_16k.zip", "SPSQUARE_16k.zip", "SCAFE_48k.zip",
                            "PSTATION_16k.zip", "PRESTO_16k.zip", "OMEETING_16k.zip", "OHALLWAY_16k.zip",
                            "NRIVER_16k.zip", "NPARK_16k.zip", "NFIELD_16k.zip", "DWASHING_16k.zip", 
                            "DLIVING_16k.zip", "DKITCHEN_16k.zip"]
        
        download_link = "https://zenodo.org/record/1227121/files/{}?download=1"
        
        for filename in noise_filenames:
            basename = os.path.splitext(filename)[0]
            self.download_dict.update({basename: download_link.format(filename)})


    def download_extract(self):
        """
        This method was overwritten because the download files were in zip format
        instead of tar format as in the base downloader class. 
        
        Limitations: 
            - this method is not idempotent as it will re-download and extract
                the files again if it is called again
        """
        save_dir = os.path.join(self.output_dir, self.dataset_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for name, url in self.download_dict.items():
            save_path = os.path.join(save_dir, name + ".zip")
            print(f"Downloading: {name}...")
            urllib.request.urlretrieve(url, filename=save_path)
            print(f"Extracting: {name}...")
            with ZipFile(save_path) as zipfile:
                zipfile.extractall(path=save_dir)
            os.remove(save_path)
            print(f"Processed: {name}")
        return save_dir


    def extract_samples(self, save_dir:str):
        """
        Extracts the wav files from the directories and copies them into the noise_dir.
        The audio files in the "SCAFE_48k" data subset are in 48 kHz and should be converted
        to 16 kHz. The if-statement in the for-loop does this conversion.

        Assumptions: 
            - The directory structure of the zip files will not change
            - 
        """
        pattern = "*/*.wav"
        high_res_audio = {"SCAFE"}
        all_wav_paths = glob.glob(os.path.join(save_dir, pattern))

        print("Extracting and removing sample files...")
        for wav_path in tqdm.tqdm(all_wav_paths):
            filename = os.path.basename(wav_path)
            dirname = os.path.basename(os.path.dirname(wav_path))
            dst_filename = "{}_{}".format(dirname, filename)
            dst_wav_path = os.path.join(self.feed_model_dir, dst_filename)
            if os.path.exists(dst_wav_path):
                print(f"{dst_wav_path} exists. Skipping...")
                continue
            else:
                # if the wavs are high resolution, down-convert to 16kHz
                if dirname in high_res_audio:
                    to_wave(wav_path, dst_wav_path)
                # if not high-res, just copy
                else: 
                    copyfile(wav_path, dst_wav_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Download voxforge dataset.")

    parser.add_argument("--output-dir",
        help="The dataset is saved in <output-dir>/<dataset-name>.")
    parser.add_argument("--dataset-name", type=str,
        help="Name of dataset with a capitalized first letter.")
    args = parser.parse_args()

    downloader = eval(args.dataset_name+"Downloader")(args.output_dir, args.dataset_name)
    print(f"type: {type(downloader)}")
    downloader.download_dataset()
