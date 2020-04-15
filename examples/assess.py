"""
this script is meant to assess a dataset along a variety of measures
author: Dustin Zubke
license: MIT
"""
# standard libary
import argparse
# third party libraries
import pandas as pd

def assess_commonvoice(validated_path:str, max_occurance:int):
    # 854445 rows total
    val_df = pd.read_csv(validated_path, delimiter='\t',encoding='utf-8')    
    accents=["us", "canada"]    
    # 231011 rows with accents "us" and "canada", 206653 with us and 24358 with canada 
    val_df = val_df[val_df.accent.isin(accents)]
    # create vote_diff column to sort the sentences upon
    val_df["vote_diff"] = val_df.up_votes - val_df.down_votes
    # remove punctiation and lower the case in sentence
    val_df['sentence']=val_df['sentence'].str.replace('[^\w\s]','').str.lower() 
    # counts of unique utterances
    val_df.sentence.value_counts(sort=True, ascending=False)
    # histogram bins
    #pd.cut(val_df.sentence.value_counts(sort=True, ascending=False),bin_range).value_counts().sort_index() 
    # dictionary of frequency counts
    count_dict=val_df.sentence.value_counts(sort=True, ascending=False).to_dict() 
    val_df = filter_by_count(val_df, count_dict, max_occurance)
    write_path=f"./validated-filtered-{max_occurance}.tsv"
    val_df.to_csv(write_path, sep="\t")

def filter_by_count(in_df:pd.DataFrame, count_dict:dict, filter_value:int):
    """
    filters the dataframe so that seteneces that occur more frequently than
    the fitler_value are reduced to a nubmer of occurances equal to the filter value,
    sentences to be filters will be done based on the difference between the up_votes and down_votes
    """
    for sentence, count in count_dict.items():
        if count> filter_value:
            # selecting rows that equal sentence
            # then sorting that subset by the vote_diff value in descending order
            # then taking the indicies of the rows after the first # of filter_values
            drop_index = in_df[in_df.sentence.eq(sentence)]\
            .sort_values("vote_diff", ascending=False)\
            .iloc[filter_value:,:].index
            
            # dropping the rows in drop_index
            in_df = in_df.drop(index=drop_index)
    return in_df
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="filters the validated.tsv file based on accent and sentence occurance")
    parser.add_argument("--validated-path", type=str,
        help="path to validated.tsv file to parse.")
    parser.add_argument("--max-occurance", type=int,
        help="max number of times a sentence can occur in output")
    args = parser.parse_args()

    assess_commonvoice(args.validated_path, args.max_occurance)
