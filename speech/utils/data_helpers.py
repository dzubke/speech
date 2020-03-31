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


def lexicon_to_dict(lexicon_path:str, corpus_name:str)->dict:
    """
    This function reads the librispeech-lexicon.txt file which is a mapping of words in the
    librispeech corpus to phoneme labels and represents the file as a dictionary.
    The digit accents are removed from the file name. 
    """
    corpus_names = ["librispeech", "tedlium", "cmudict", "common-voice"]
    if corpus_name not in corpus_names:
        raise ValueError("corpus_name not accepted")
    
    lex_dict = defaultdict(lambda: UNK_WORD_TOKEN)
    with open(lexicon_path, 'r', encoding="ISO-8859-1") as fid:
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

    if corpus_name == "librispeech" or corpus_name == "cmudict":
        return list(map(lambda x: x.rstrip(string.digits), phonemes))
    else:
        return phonemes


def clean_dict(lex_dict, corpus_name):
    
    if corpus_name == "tedlium" or corpus_name =="cmudict":
        return defaultdict(lambda: UNK_WORD_TOKEN, 
                {key: value for key, value in lex_dict.items() if not re.search("\(\d\)$", key)})
    else: 
        return lex_dict


class UnknownWords():

    def __init__(self):
        self.word_set:set = set()
        self.filename_dict:dict = dict()
        self.line_count:int = 0
        self.word_count:int = 0
        self.has_unknown= False


    def check_transcript(self, filename:str, text, word_phoneme_dict:dict):

        if type(text) == str:
            text = text.split()
        elif type(text) == list: 
            pass
        else: 
            raise(TypeError("input text is not string or list type"))
   
        self.line_count += 1
        self.word_count += len(text) - 1
        line_unk = [word for word in text if word_phoneme_dict[word]==UNK_WORD_TOKEN]
        #if line_unk is empty, has_unk is False
        self.has_unknown = bool(line_unk)
        if self.has_unknown:
            self.word_set.update(line_unk)
            self.filename_dict.update({filename: len(line_unk)})

    def process_save(self, save_path:str):
        """
        saves a json object of the dictionary with relevant statistics on the unknown words in corpus
        """

        stats_dict=dict()
        stats_dict.update({"unique_unknown_words": len(self.word_set)})
        stats_dict.update({"count_unknown_words": sum(self.filename_dict.values())})
        stats_dict.update({"total_words": self.word_count})
        stats_dict.update({"lines_unknown_words": len(self.filename_dict)})
        stats_dict.update({"total_lines": self.line_count})
        stats_dict.update({"unknown_words_set": list(self.word_set)})
        stats_dict.update({"unknown_words_dict": self.filename_dict})
        
        stats_dir = "unk_word_stats"
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)

        stats_dict_fname = os.path.join(stats_dir, os.path.basename(save_path)+"_unk-words-stats.json")
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


def combine_lexicons(lex1_dict:dict, lex2_dict:dict)->(dict, dict):
    """
    this function takes as input a dictionary representation of the two
    lexicons and outputs a combined dictionary lexicon. it also outputs
    a dict of words with different pronunciations
    Arguments:
        lex1_dict - dict[str:list(str)]: dict representation of the first lexicon
        lex2_dict - dict[str:list(str)]: dict representation of the second lexicon
    Returns:
        combo_dict - dict[str:list(str)]
    """

    word_set = set(list(lex1_dict.keys()) + list(lex2_dict.keys()))
    combo_dict = defaultdict(lambda: list())
    diff_labels = dict()

    for word in word_set:
        if  word not in lex1_dict:
            # word has to be in lex2_dict
            combo_dict.update({word:lex2_dict.get(word)})
        elif word not in lex2_dict:
            # word has to be in lex1_dict
            combo_dict.update({word:lex1_dict.get(word)})
        else:
            # word is in both dicts, used lex2_dict
            if lex1_dict.get(word) == lex2_dict.get(word):
                combo_dict.update({word:lex2_dict.get(word)})
            else:   # phoneme labels are not the same
                combo_dict.update({word:lex2_dict.get(word)})
                diff_labels.update({word: {"lex1": lex1_dict.get(word), "lex2": lex2_dict.get(word)}})
    # print(f"words with different phoneme labels are: \n {diff_labels}")
    print(f"number of words with different labels: {len(diff_labels)}")

    return  combo_dict, diff_labels


def create_lexicon(cmu_dict:dict, ted_dict:dict, lib_dict:dict, out_path:str='')->dict:
    """
    Creates a master lexicon using pronuciations from first cmudict, then tedlium
    dictionary and finally librispeech. 
    Arguments:
        cmu_dict - dict[str:list(str)]: cmu dict processed with lexicon_to_dict
        ted_dict - dict[str:list(str)]: tedlium dict processed with lexicon_to_dict
        lib_dict - dict[str:list(str)]: librispeech dict processed with lexicon_to_dict
        out_path - str (optional): output path where the master lexicon will be written to
    Returns:
        master_dict - dict[str:list(str)]
    """

    word_set = set(list(cmu_dict.keys()) + list(ted_dict.keys())+list(lib_dict.keys()))
    master_dict = defaultdict(lambda: UNK_WORD_TOKEN)

    # uses the cmu_dict pronunciation first, then tedlium_dict, and last librispeech_dict
    for word in word_set:
        if  word in cmu_dict:
            master_dict.update({word:cmu_dict.get(word)})
        elif word in ted_dict:
            master_dict.update({word:ted_dict.get(word)})
        elif word in lib_dict:
            master_dict.update({word:lib_dict.get(word)})

    if out_path != '': 
        sorted_keys = sorted(master_dict.keys())
        with open(out_path, 'w') as fid:
            for key in sorted_keys:
                fid.write(f"{key} {' '.join(master_dict.get(key))}\n")
 
    return master_dict