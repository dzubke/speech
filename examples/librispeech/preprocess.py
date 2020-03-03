from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import json
import os
import tqdm
import wave
import sys
from collections import defaultdict
import pickle
import string

from speech.utils import data_helpers
from speech.utils import wave


def main(output_directory, use_phonemes):
    # "train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "dev-other", "test-clean", "dev-other"  
    SETS = {
    "train" : [],
    "dev" : [],
    "test" : ["test-clean"],
    }

    path = os.path.join(output_directory, "LibriSpeech")   
    print("Converting files from flac to wave...")
    #convert_to_wav(path)
    
    for dataset, dirs in SETS.items():
        for d in dirs:
            print("Preprocessing {}".format(d))
            prefix = os.path.join(path, d)
            build_json(prefix, use_phonemes)


def build_json(path, use_phonemes):
    transcripts = load_transcripts(path) #, unk_words_set, unk_words_dict, line_count, word_count
    dirname = os.path.dirname(path)
    basename = os.path.basename(path) + os.path.extsep + "json"
    unknown_set=set()
    unknown_dict=dict()
    line_count, word_count= 0, 0

    if use_phonemes: 
        LEXICON_PATH = "librispeech-lexicon_extended.txt"
        word_phoneme_dict = data_helpers.lexicon_to_dict(LEXICON_PATH, corpus_name="librispeech")
    with open(os.path.join(dirname, basename), 'w') as fid:
        for file_key, text in tqdm.tqdm(transcripts.items()):
            wave_file = path_from_key(file_key, path, ext="wav")
            dur = wave.wav_duration(wave_file)

            if use_phonemes: 
                unk_words_list, unk_words_dict, counts = data_helpers.check_unknown_words(file_key, text, word_phoneme_dict)
                if counts[1] > 0: 
                    unknown_set.update(unk_words_list)
                    unknown_dict.update(unk_words_dict)
                    line_count+=counts[0]
                    word_count+=counts[1]
                    continue
                text = transcript_to_phonemes(text, word_phoneme_dict)
                    
            datum = {'text' : text,
                     'duration' : dur,
                     'audio' : wave_file}
            json.dump(datum, fid)
            fid.write("\n")

    process_unknown_words(path, unknown_set, unknown_dict, line_count, word_count)
    

def convert_to_wav(path):
    data_helpers.convert_full_set(path, "*/*/*/*.flac")


def load_transcripts(path):
    pattern = os.path.join(path, "*/*/*.trans.txt")
    files = glob.glob(pattern)
    data = {}
    # unknown_set=set()
    # unknown_dict=dict()
    # line_count, word_count= 0, 0

    # if use_phonemes: 
    #     LEXICON_PATH = "librispeech-lexicon.txt"
    #     word_phoneme_dict = data_helpers.lexicon_to_dict(LEXICON_PATH, corpus_name="librispeech")
    for f in tqdm.tqdm(files):
        with open(f) as fid:
            # load transcript of file
            lines = [l.strip().lower().split() for l in fid]
            # if use_phonemes: 
            #     file_unk_list, file_unk_dict, counts = check_unknown_words(lines, word_phoneme_dict)
            #     lines = ((l[0], l[1:], word_phoneme_dict)) for l in lines)
            #     unknown_set.update(file_unk_list)
            #     unknown_dict.update(file_unk_dict)
            #     line_count+=counts[0]
            #     word_count+=counts[1]

            # else: 
            lines = ((l[0], " ".join(l[1:])) for l in lines)
                # unk_words = []
            data.update(lines)
    return data #, unknown_set, unknown_dict, line_count, word_count



def transcript_to_phonemes(words, word_phoneme_dict):
    """converts the words in the transcript to phonemes using the word_to_phoneme dictionary mapping
    """
    phonemes = []
    for word in words:
        phonemes.extend(word_phoneme_dict[word])
    return phonemes

def path_from_key(key, prefix, ext):
    dirs = key.split("-")
    dirs[-1] = key
    path = os.path.join(prefix, *dirs)
    return path + os.path.extsep + ext


def process_unknown_words(path, unknown_words_set, unknown_words_dict, line_count, word_count):
    """saves a json object of the dictionary with relevant statistics on the unknown words in corpus
    """

    stats_dict=dict()
    stats_dict.update({"unique_unknown_words": len(unknown_words_set)})
    stats_dict.update({"count_unknown_words": sum(unknown_words_dict.values())})
    stats_dict.update({"total_words": word_count})
    stats_dict.update({"lines_unknown_words": len(unknown_words_dict)})
    stats_dict.update({"total_lines": line_count})
    stats_dict.update({"unknown_words_set": list(unknown_words_set)})
    stats_dict.update({"unknown_words_dict": unknown_words_dict})

    stats_dict_fname = "libsp_"+os.path.basename(path)+"_unk-words-stats.json"
    with open(stats_dict_fname, 'w') as fid:
        json.dump(stats_dict, fid)


def unique_unknown_words():
    """
        Creates a set of the total number of unknown words across the 7 segments of the librispeech dataset
    """

    train_100_fn = 'libsp_train-clean-100_unk-words-stats.json'
    train_360_fn = 'libsp_train-clean-360_unk-words-stats.json'
    train_500_fn = 'libsp_train-other-500_unk-words-stats.json'
    test_clean_fn = 'libsp_test-clean_unk-words-stats.json'
    test_other_fn = 'libsp_test-other_unk-words-stats.json'
    dev_clean_fn = 'libsp_dev-clean_unk-words-stats.json'
    dev_other_fn = 'libsp_dev-other_unk-words-stats.json'

    datasets_fn = [train_100_fn, train_360_fn, train_500_fn, test_clean_fn, test_other_fn, dev_clean_fn, dev_other_fn]
    unknown_set = set()
    for data_fn in datasets_fn: 
        with open(data_fn, 'r') as fid: 
            unk_words_dict = json.load(fid)
            unknown_set.update(unk_words_dict['unknown_words_set'])
            print(len(unk_words_dict['unknown_words_set']))

    unknown_set = list(filter(lambda x: len(x)<30, unknown_set))
    
    with open("libsp_all_unk_words.txt", 'w') as fid:
        fid.write('\n'.join(unknown_set))

    print(f"number of unknown words: {len(unknown_set)}")

if __name__ == "__main__":
    ## format of command is >>python preprocess.py <path_to_dataset> --use_phonemes <True/False> 
    # where the optional --use_phonemes argument is whether the labels will be phonemes (True) or words (False)
    parser = argparse.ArgumentParser(
            description="Preprocess librispeech dataset.")

    parser.add_argument("output_directory",
        help="The dataset is saved in <output_directory>/LibriSpeech.")

    parser.add_argument("--use_phonemes",
        help="A boolean of whether the labels will be phonemes (True) or words (False)")
    args = parser.parse_args()

    main(args.output_directory, args.use_phonemes)
