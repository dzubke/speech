from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import tqdm
from collections import defaultdict
import string
import re

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
    """This function reads the librispeech-lexicon.txt file which is a mapping of words in the
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


def check_unknown_words(filename, text, word_phoneme_dict):
    unk_words_list, unk_words_dict = list(), dict()
    line_count, word_count = 0, 0
    text = text.split()
    line_count += 1
    word_count += len(text) - 1
    line_unk_list = [word for word in text if word_phoneme_dict[word]==UNK_WORD_TOKEN]
    if line_unk_list:       #if not empty
        unk_words_list.extend(line_unk_list)
        unk_words_dict.update({filename: len(line_unk_list)})

    return unk_words_list, unk_words_dict, (line_count, word_count)


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

