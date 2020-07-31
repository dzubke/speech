
# standard libraries
import argparse
from datetime import date
import os
import random

# project libraries
from speech.utils.io import read_data_json, write_data_json

def main(dataset_path:str, write_path: str, subset_size:int, use_internal:bool, mix_with_speak:bool):
    """
    Arguments
    ----------
    use_internal - bool: if True, the internal dataset paths set within main()will be used instead of input args
    mix_with_speak - bool: if True, the datasets will be mixed with the speak testset

    """

    if not use_internal:
        subsetor = DataSubsetor(dataset_path, int(subset_size))
        subsetor.write_subset(write_path)
    else: 
        data_name_path= {
            "cv-train-dd": "/home/dzubke/awni_speech/data/common-voice/v5_2020-06-22/validated-50-maxrepeat_dd.json",
            "libsp-train-dd": "/home/dzubke/awni_speech/data/LibriSpeech/train-other-960_dd.json",
            "ted-train-dd": "/home/dzubke/awni_speech/data/tedlium/TEDLIUM_release-3/train_dd.json",
        }
        # subset_size = 100
        today_date = str(date.today())
        dirname = "/home/dzubke/awni_speech/data/subsets/20200730/"
        os.makedirs(dirname, exist_ok=True)
        write_path_str = dirname + "{name}_{size}_{date}.json"
        mix_write_path_str = dirname + "speak_{name}_{size}_{date}.json" 

        for data_name, data_path in data_name_path.items():
            # samples from only one dataset
            subsetor = DataSubsetor(data_path, subset_size)
            write_path = write_path_str.format(name=data_name, size=subset_size, date=today_date)
            subsetor.write_subset(write_path)

            if mix_with_speak:
                # samples from datasets mixed with speak tests data
                speak_path = "/home/dzubke/awni_speech/data/speak_test_data/2020-05-27/speak-test_2020-05-27.json"
                speak_subsetor = DataSubsetor(speak_path, 91)            
                mix_write_path = mix_write_path_str.format(name=data_name, size=subset_size, date=today_date)
                write_mixed_subset(subsetor.get_subset(), speak_subsetor.get_subset(), mix_write_path)

class DataSubsetor():
    def __init__(self, dataset_path:str, subset_size:int):
        self.dataset_path = dataset_path
        self.data_json = read_data_json(dataset_path)
        self.subset = random.sample(self.data_json, k=subset_size)        
    
    def get_full_dataset(self):
        return self.data_json

    def get_subset(self):
        return self.subset

    def write_subset(self, write_path:str):
        write_data_json(self.subset, write_path)

def write_mixed_subset(subset_1:list, subset_2:list,  write_path:str):
    subset_size = len(subset_1)
    mixed_subset = subset_1[:subset_size//2] + subset_2[:subset_size//2]        
    write_data_json(mixed_subset, write_path)







if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Creates a subset of a dataset and writes a new dataset to the write_path.")
    parser.add_argument("--dataset-path",
        help="The path to the dataset to be subsetted.")
    parser.add_argument("--subset-size", type=int,
        help="Number of samples in the data subset.")
    parser.add_argument("--write-path",
        help="Path where to write the subset.")
    parser.add_argument("--use-internal", action="store_true", default=False,
        help="Uses internal values set within main() instead of input args.")
    parser.add_argument("--mix-with-speak", action="store_true", default=False,
        help="If true, the datasets will be mixed with the speak testset")
    ARGS = parser.parse_args()


    print("mix with speak: ", ARGS.mix_with_speak)
    main(ARGS.dataset_path, ARGS.write_path, ARGS.subset_size, ARGS.use_internal, ARGS.mix_with_speak)
