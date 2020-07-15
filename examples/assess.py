"""
this script is meant to assess a dataset along a variety of measures
author: Dustin Zubke
license: MIT
"""
# standard libary
import argparse
# third party libraries
import pandas as pd
# project libraries
from speech.dataset_info import AllDatasets, TatoebaDataset
from speech.utils import wave, data_helpers


def assess_commonvoice(validated_path:str, max_occurance:int):

    val_df = pd.read_csv(validated_path, delimiter='\t',encoding='utf-8')
    print(f"there are {val_df.shape[0]} entries/rows in the dataset")
    accents=["us", "canada"]    
    # 231011 rows with accents "us" and "canada", 206653 with us and 24358 with canada 
    val_df = val_df[val_df.accent.isin(accents)]
    print(f"there are {val_df.shape[0]} entries with accents {accents}")
    # create vote_diff column to sort the sentences upon
    val_df["vote_diff"] = val_df.up_votes - val_df.down_votes
    # remove punctiation and lower the case in sentence
    val_df['sentence']=val_df['sentence'].str.replace('[^\w\s]','').str.lower() 
    # sorts by the number of unique utterances in descending order
    val_df.sentence.value_counts(sort=True, ascending=False)
    # histogram bins
    #pd.cut(val_df.sentence.value_counts(sort=True, ascending=False),bin_range).value_counts().sort_index() 
    # dictionary of frequency counts
    count_dict=val_df.sentence.value_counts(sort=True, ascending=False).to_dict() 
    # filters so utterances only have at most max_occurances
    # keeps utterances with highest difference between up_votes and down_votes
    val_df, drop_row_count = filter_by_count(val_df, count_dict, max_occurance)
    print(f"number of rows dropped: {drop_row_count}")
    dirname = os.path.dirname(validated_path)
    write_path = os.path.join(dirname, f"validated-filtered-{max_occurance}.tsv")
    if os.path.exists(write_path):
        print(f"file: {write_path} already exists.")
        print("Would you like to rewrite it? y/n")
        answer = input()
        if answer in ["Y", "y"]:
            val_df.to_csv(write_path, sep="\t", index=False)
        else: 
            print("file has not be overwritten. No new file saved")
    else:
        val_df.to_csv(write_path, sep="\t", index=False)



    write_path=

def filter_by_count(in_df:pd.DataFrame, count_dict:dict, filter_value:int):
    """
    filters the dataframe so that seteneces that occur more frequently than
    the fitler_value are reduced to a nubmer of occurances equal to the filter value,
    sentences to be filters will be done based on the difference between the up_votes and down_votes
    """
    drop_row_count = 0 
    for sentence, count in count_dict.items():
        if count > filter_value:
            # selecting rows that equal sentence
            # then sorting that subset by the vote_diff value in descending order
            # then taking the indicies of the rows after the first # of filter_values
            drop_index = in_df[in_df.sentence.eq(sentence)]\
            .sort_values("vote_diff", ascending=False)\
            .iloc[filter_value:,:].index
            
            drop_row_count += len(drop_index)
            # dropping the rows in drop_index
            in_df = in_df.drop(index=drop_index)
    return in_df, drop_row_count


class DurationAssessor():

    def __init__(self):
        self.datasets = AllDatasets().dataset_list

    def duration_report(self, save_path:str):
        with open(save_path, 'w') as fid:
            for dataset in self.datasets:
                duration = dataset.get_duration()
                name = str(type(dataset))
                out_string = "{0}: {1}\n".format(name, duration)
                fid.write(out_string)
            



class TatoebaAssessor():

    def __init__(self):
        self.dataset = TatoebaDataset()  

    def create_report(self):
        raise NotImplementedError
    
    def audio_by_speaker(self):
        assess_dict = dict()
        audio_files = self.dataset.get_audio_files()



    

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="filters the validated.tsv file based on accent and sentence occurance")
    parser.add_argument("--validated-path", type=str,
        help="path to validated.tsv file to parse.")
    parser.add_argument("--max-occurance", type=int,
        help="max number of times a sentence can occur in output")
    args = parser.parse_args()

    assess_commonvoice(args.validated_path, args.max_occurance)
