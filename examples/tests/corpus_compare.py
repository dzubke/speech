# standard libraries
import json
import os
import argparse
from collections import defaultdict
# project libraries
from speech.utils import data_helpers

def main(corpus1_lex_fn:str, corpus2_lex_fn:str):
    """
    performs a variety of methods comparing the lexicon pronounciation
    dictionaries of corpus1 and corpus2
    Arguments:
        corpus1_lex_fn (str): the pathname to the pronunciation
            lexicon for corpus 1
        corpus2_lex_fn (str): the pathname to the pronunciation
            lexicon for corpus 2
    """


    corpus1_name = process_name(corpus1_lex_fn)
    corpus2_name = process_name(corpus2_lex_fn)
    
    lex1_dict = data_helpers.lexicon_to_dict(corpus1_lex_fn, corpus1_name)
    lex2_dict = data_helpers.lexicon_to_dict(corpus2_lex_fn, corpus2_name)

    compare_phones(lex1_dict, corpus1_name, lex2_dict, corpus2_name)

def process_name(corpus_lex_fn:str):
    """
    extracts the corpus name from the lexicon dict filename
    """
    corpus_name = os.path.basename(corpus_lex_fn)
    if '-' in corpus_name: 
        #for librispeech-lexicon.txt
        corpus_name = corpus_name.split(sep='-')[0]
    else:
        #for TEDLIUM.152k.dic
        corpus_name = corpus_name.split(sep='.')[0].lower()
    return corpus_name


def compare_phones(corpus1_lex:str, corpus1_name:str,  corpus2_lex:str, corpus2_name:str):
    """
    compares the phonemes in lexicons of corpus1 and corpus2 and
    creates
    """

    corpus1_phones, corpus1_words = export_phones(corpus1_lex, export=False)
    corpus2_phones, corpus2_words = export_phones(corpus2_lex, export=False)
    intersection = corpus2_phones.intersection(corpus1_phones)
    print(f"phonemes in {corpus1_name} but not in {corpus2_name}: \
        {corpus1_phones.difference(corpus2_phones)} ")
    print(f"phonemes in {corpus2_name} but not in {corpus1_name}: \
        {corpus2_phones.difference(corpus1_phones)} ")
    print(f"phonemes in common: number: {len(intersection)}, phonemes: {intersection}")


def export_phones(lex_dict:dict, export:bool=False):
    """
    exports all words and phones in lex_dict into two separate txt files
    Arguments:
        lex_dict (dict[str:list]): the pronuncation dictionary that maps words to a list of phoneme labels
        export (bool): a boolean that if True will export the list of phoneme labels to a txt file. 
    """

    phone_set = set()
    word_set = set()
    for word, phones in lex_dict.items():
        if word not in word_set:
            word_set.add(word)
        
        for phone in phones:
            if phone not in phone_set:
                phone_set.add(phone)

    if export: 
        phones_filename = "lexicon_phone_set.txt"
        words_filename = "lexicon_word_set.txt"
        with open(phones_filename, 'w') as fid:
            for phone in phone_set:
                fid.write(phone+'\n')

        with open(words_filename, 'w') as fid:
            for word in word_set:
                fid.write(word+'\n')

    return phone_set, word_set


def check_phones():
    """
    This function compares the phonemes in the librispeech corpus with the phoneme labels in the 39-phonemes
    in the timit dataset outlined here: 
    https://www.semanticscholar.org/paper/Speaker-independent-phone-recognition-using-hidden-Lee-Hon/3034afcd45fc190ed71982828b77f6e4154bdc5c
    Discrepencies in the CMU-39 and timit-39 phoneme sets and the librispeech phonemes: 
     - included in CMU-39 but not timit-39:  ao, zh, 
     - included timit-39 but not CMU-39: dx, sil
    """
    # standard 39 phones in the timit used by awni dictionary
    timit_phones39 = set(['ae', 'ah', 'aa', 'aw', 'er', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'l', 'm', 'n', 'ng', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'sil'])
    cmu_phones = set(['aa', 'ae', 'ah', 'ao', 'aw', 'ay',  'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh'])
    print(f"length of timit_dict: {len(timit_phones39)}")
    librispeech_phones = set()
    
    # greating a set of the librispeech phones by looping over every phone list in the word_to_phoneme mapping
    for phones in word_phoneme_dict.values():
        # looping over every phone in the word pronunciation
        for phone in phones:
            if phone not in librispeech_phones:
                librispeech_phones.add(phone)

    print(f"phones in librispeech but not cmu: {librispeech_phones.difference(cmu_phones)}")
    print(f"phones in cmu but not librispeech: {cmu_phones.difference(librispeech_phones)}")
    print(f"phones in timit but not cmu: {timit_phones39.difference(cmu_phones)}")
    print(f"phones in cmubut not timit: {cmu_phones.difference(timit_phones39)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Preprocess librispeech dataset.")

    parser.add_argument("corpus1_lex_fn",
        help="The path to the lexicon pronuciation dictionary for corpus 1.")

    parser.add_argument("corpus2_lex_fn",
        help="The path to the lexicon pronuciation dictionary for corpus 2")
    args = parser.parse_args()

    main(args.corpus1_lex_fn, args.corpus2_lex_fn)
