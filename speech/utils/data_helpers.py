from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import tqdm
from collections import defaultdict
import string
import re
import json

from speech.utils import convert

UNK_WORD_TOKEN = list()

def convert_full_set(path, pattern, new_ext="wav", **kwargs):
    pattern = os.path.join(path, pattern)
    audio_files = glob.glob(pattern)
    for af in tqdm.tqdm(audio_files):
        base, ext = os.path.splitext(af)
        wav = base + os.path.extsep + new_ext
        convert.to_wave(af, wav, **kwargs)


def lexicon_to_dict(lexicon_path, corpus_name):
    """
        This function reads the librispeech-lexicon.txt file which is a mapping of words in the
        librispeech corpus to phoneme labels and represents the file as a dictionary.
        The digit accents are removed from the file name. 
        Note: the librispeech-lexicon.txt file needs to be in the same directory as this file.
    """
    
    lex_dict = defaultdict(lambda: UNK_WORD_TOKEN)
    with open(lexicon_path, 'r') as fid:
        lexicon = (l.strip().lower().split() for l in fid)
        for line in lexicon: 
            word = line[0]
            phones = line[1:]
            phones = clean_phonemes(phones, corpus_name)
            # librispeech: the if-statement will ignore the second pronunciation with the same word
            if lex_dict[word] == UNK_WORD_TOKEN:
                lex_dict[word] = phones
    lex_dict = clean_dict(lex_dict, corpus_name)
    assert type(lex_dict)== defaultdict, "word_phoneme_dict is not defaultdict"
    return lex_dict


def clean_phonemes(phonemes, corpus_name):

    if corpus_name == 'librispeech':
        return list(map(lambda x: x.rstrip(string.digits), phonemes))
    else:
        return phonemes


def clean_dict(lex_dict, corpus_name):
    
    if corpus_name == "tedlium":
        return defaultdict(lambda: UNK_WORD_TOKEN, 
                {key: value for key, value in lex_dict.items() if not re.search("\(\d\)$", key)})
    else: 
        return lex_dict


def check_unknown_words(filename, text, word_phoneme_dict):

    if type(text) == str:
        text = text.split()
    elif type(text) == list: 
        pass
    else: 
        raise(TypeError("input text is not string or list type"))
    unk_words_list, unk_words_dict = list(), dict()
    line_count, word_count = 0, 0
    line_count += 1
    word_count += len(text) - 1
    line_unk_list = [word for word in text if word_phoneme_dict[word]==UNK_WORD_TOKEN]
    if line_unk_list:       #if not empty
        unk_words_list.extend(line_unk_list)
        unk_words_dict.update({filename: len(line_unk_list)})

    return unk_words_list, unk_words_dict, (line_count, word_count)


def process_unknown_words(path, unknown_words_set, unknown_words_dict, line_count, word_count):
    """
        saves a json object of the dictionary with relevant statistics on the unknown words in corpus
    """

    stats_dict=dict()
    stats_dict.update({"unique_unknown_words": len(unknown_words_set)})
    stats_dict.update({"count_unknown_words": sum(unknown_words_dict.values())})
    stats_dict.update({"total_words": word_count})
    stats_dict.update({"lines_unknown_words": len(unknown_words_dict)})
    stats_dict.update({"total_lines": line_count})
    stats_dict.update({"unknown_words_set": list(unknown_words_set)})
    stats_dict.update({"unknown_words_dict": unknown_words_dict})
    
    stats_dir = "unk_word_stats"
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    stats_dict_fname = os.path.join(stats_dir, os.path.basename(path)+"_unk-words-stats.json")
    with open(stats_dict_fname, 'w') as fid:
        json.dump(stats_dict, fid)


def unique_unknown_words(dataset_dir):
    """
        Creates a set of the total number of unknown words across all segments in a dataset assuming a
        unk-words-stats.json file from process_unknown_words() has been created for each part of the dataset. 

        Arguments:
            dataset_dir (str): pathname of dir continaing "unknown_word_stats" dir with unk-words-stats.json files
    """

    pattern = os.path.join(dataset_dir, "unk_word_stats", "*unk-words-stats.json")
    dataset_list = glob.glob(pattern)
    if len(dataset_list) == 0: 
        train_100_fn = './unk_word_stats/libsp_train-clean-100_unk-words-stats.json'
        train_360_fn = './unk_word_stats/libsp_train-clean-360_unk-words-stats.json'
        train_500_fn = './unk_word_stats/libsp_train-other-500_unk-words-stats.json'
        test_clean_fn = './unk_word_stats/libsp_test-clean_unk-words-stats.json'
        test_other_fn = './unk_word_stats/libsp_test-other_unk-words-stats.json'
        dev_clean_fn = './unk_word_stats/libsp_dev-clean_unk-words-stats.json'
        dev_other_fn = './unk_word_stats/libsp_dev-other_unk-words-stats.json'
        dataset_list = [train_100_fn, train_360_fn, train_500_fn, test_clean_fn, test_other_fn, dev_clean_fn, dev_other_fn]
    
    unknown_set = set()
    for data_fn in dataset_list: 
        with open(data_fn, 'r') as fid: 
            unk_words_dict = json.load(fid)
            unknown_set.update(unk_words_dict['unknown_words_set'])
            print(len(unk_words_dict['unknown_words_set']))


    unknown_set = filter_set(unknown_set)
    unknown_list = list(unknown_set)
    
    write_path = os.path.join(dataset_dir, "unk_word_stats","all_unk_words.txt")
    with open(write_path, 'w') as fid:
        fid.write('\n'.join(unknown_list))

    print(f"number of filtered unknown words: {len(unknown_list)}")


def filter_set(unknown_set:set):
    """
        filters the set based on the length and presence of digits.
    """
    unk_filter = filter(lambda x: len(x)<30, unknown_set)
    search_pattern = r'[0-9!#$%&()*+,\-./:;<=>?@\[\\\]^_{|}~]'
    unknown_set = set(filter(lambda x: not re.search(search_pattern, x), unk_filter))
    return unknown_set
