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
    
    SETS = {
    "train" : ["train-other-500"],
    "dev" : ["dev-other"],
    "test" : ["test-other"],
    }

    path = os.path.join(output_directory, "LibriSpeech")   
    print("Converting files from flac to wave...")
    convert_to_wav(path)
    
    for dataset, dirs in SETS.items():
        for d in dirs:
            print("Preprocessing {}".format(d))
            prefix = os.path.join(path, d)
            build_json(prefix, use_phonemes)


def build_json(path, use_phonemes):
    transcripts = load_transcripts(path, use_phonemes)
    dirname = os.path.dirname(path)
    basename = os.path.basename(path) + os.path.extsep + "json"
    with open(os.path.join(dirname, basename), 'w') as fid:
        for file_key, text in tqdm.tqdm(transcripts.items()):
            wave_file = path_from_key(file_key, path, ext="wav")
            dur = wave.wav_duration(wave_file)
            datum = {'text' : text,
                     'duration' : dur,
                     'audio' : wave_file}
            json.dump(datum, fid)
            fid.write("\n")


def load_transcripts(path, use_phonemes=True):
    pattern = os.path.join(path, "*/*/*.trans.txt")
    files = glob.glob(pattern)
    data = {}
    if use_phonemes: 
        word_phoneme_dict = lexicon_to_dict()
        #save_lexicon_to_dict()

    for f in tqdm.tqdm(files):
        with open(f) as fid:
            lines = (l.strip().lower().split() for l in fid)
            if use_phonemes: 
                lines = (
                    (l[0], phones for word in l[1:] 
                                    for phones in word_phoneme_dict[word]) 
                                        for l in lines)
            else: 
                lines = ((l[0], " ".join(l[1:])) for l in lines)
            data.update(lines)
    return data


def transcript_to_phonemes(words):
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


def convert_to_wav(path):
    data_helpers.convert_full_set(path, "*/*/*/*.flac")


def lexicon_to_dict():
    """This function reads the librispeech-lexicon.txt file which is a mapping of words in the
        librispeech corpus to phoneme labels and represents the file as a dictionary.
        The digit accents are removed from the file name. 
        Note: the librispeech-lexicon.txt file needs to be in the same directory as this file.
    """
    
    lex_dict = defaultdict(lambda: "unk")
    with open("librispeech-lexicon.txt", 'r') as fid:
        lexicon = (l.strip().lower().split() for l in fid)
        for line in lexicon: 
            word = line[0]
            phones = line[1:]
            # remove the accent digit from the phone, string.digits = '0123456789'
            phones = list(map(lambda x: x.rstrip(string.digits), phones))
            # the if-statement will ignore the second pronunciation (phone list)
            if lex_dict[word] == "unk":
                lex_dict[word] = phones

    return lex_dict


def check_phones():
    """This function compares the phonemes in the librispeech corpus with the phoneme labels in the 39-phonemes
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