
# standard libraries
import argparse
from datetime import date
import random

# project libraries
from speech.utils.io import read_data_json, write_data_json

def main(dataset_path:str, write_path: str, subset_size:int, use_internal:bool):
    """
    If the use_internal arguement is True, the else loop will create and subset the given
    datasets
    """

    if not use_internal:
        subsetor = DataSubsetor(dataset_path)
        write_subset(subsetor.data_json, write_path, subset_size)
    else: 
        data_name_path= {
            "cv_dev": "/home/dzubke/awni_speech/data/common-voice/dev.json",
            "libsp_dev": "/home/dzubke/awni_speech/data/LibriSpeech/dev-combo.json",
            "ted_dev": "/home/dzubke/awni_speech/data/tedlium/TEDLIUM_release-3/dev.json",
        }
        subset_size = 100
        today_date = str(date.today())
        write_path_str = "/home/dzubke/awni_speech/data/subsets/20200603/{name}_{size}_{date}.json"

        # samples from only one dataset
        for data_name, data_path in data_name_path:
            subsetor = DataSubsetor(data_path)
            write_path = write_path_str.format(data_name, subset_size, today_date)
            write_subset(subsetor.get_data_list(), subset_size, write_path)

        

class DataSubsetor():
    def __init__(self, dataset_path:str):
        self.dataset_path = dataset_path
        self.data_json = read_data_json(dataset_path)
    
    def get_data_list(self):
        return self.data_json

    
def write_subset(data_json:list, subset_size:int=100, write_path:str):
        subset = random.sample(data_json, k=subset_size)
        write_data_json(subset, write_path)



# def mix_data_subsets(subsetor_1:DataSubsetor, subsetor_2:DataSubsetor, subset_size:int, write_path:str):






if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Creates a subset of a dataset and writes a new dataset to the write_path.")
    parser.add_argument("--dataset-path",
        help="The path to the dataset to be subsetted.")
    parser.add_argument("--subset-size",
        help="Number of samples in the data subset.")
    parser.add_argument("--write-path",
        help="Path where to write the subset.")
    parser.add_argument("--use-internal", action="store_true", default=False,
        help="Uses internal values set within main() instead of input args.")
    ARGS = parser.parse_args()



    main(ARGS.dataset_path, ARGS.write_path, ARGS.subset_size, ARGS.use_internal)