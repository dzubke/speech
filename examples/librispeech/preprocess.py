from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import json
import os
import tqdm
import wave
from collections import defaultdict
import string

from speech.utils import data_helpers
from speech.utils import wave

SETS = {
    "train" : ["train-clean-100"],
    "dev" : ["dev-clean"],
    "test" : ["test-clean"]
    }


def load_transcripts(path, use_phonemes=True):
    pattern = os.path.join(path, "*/*/*.trans.txt")
    files = glob.glob(pattern)
    data = {}
    for f in tqdm.tqdm(files):
        with open(f) as fid:
            lines = (l.strip().lower().split() for l in fid)
            if use_phonemes: 
                lines = ((l[0], transcript_to_phonemes(l[1:])) for l in lines)
            else: 
                lines = ((l[0], " ".join(l[1:])) for l in lines)
            data.update(lines)
    return data

def path_from_key(key, prefix, ext):
    dirs = key.split("-")
    dirs[-1] = key
    path = os.path.join(prefix, *dirs)
    return path + os.path.extsep + ext

def convert_to_wav(path):
    data_helpers.convert_full_set(path, "*/*/*/*.flac")

def clean_text(text):
    return text.strip().lower()

def build_json(path, use_phonemes):
    transcripts = load_transcripts(path, use_phonemes)
    dirname = os.path.dirname(path)
    basename = os.path.basename(path) + os.path.extsep + "json"
    with open(os.path.join(dirname, basename), 'w') as fid:
        for k, t in tqdm.tqdm(transcripts.items()):
            wave_file = path_from_key(k, path, ext="wav")
            dur = wave.wav_duration(wave_file)
            datum = {'text' : t,
                     'duration' : dur,
                     'audio' : wave_file}
            json.dump(datum, fid)
            fid.write("\n")


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
            striped_phones = []
            for phone in phones:
                striped_phones.append(phone.rstrip(string.digits))
            lex_dict[word] = striped_phones

    return lex_dict

# creating a global instance of the word_to_phoneme dictionary
word_to_phoneme = lexicon_to_dict()

def transcript_to_phonemes(words):
    """converts the words in the transcript to phonemes using the word_to_phoneme dictionary mapping
    """
    phonemes = []
    for word in words:
        phonemes.extend(word_to_phoneme[word])

    return phonemes


def check_phones():
    """This function compares the phonemes in the librispeech corpus with the phoneme labels in the 39-phonemes
    in the timit dataset outlined here: 
    https://www.semanticscholar.org/paper/Speaker-independent-phone-recognition-using-hidden-Lee-Hon/3034afcd45fc190ed71982828b77f6e4154bdc5c
    
    Discrepencies in the CMU-39 and timit-39 phoneme sets and the librispeech phonemes: 
     - included in CMU-39 but not timit-39:  ao, zh, 
     - included timit-39 but not CMU-39: dx, sil
     - not included in CMU or timit:  u
    """
    # standard 39 phones in the timit used by awni dictionary
    awni_dct39 = set(['ae', 'ah', 'aa', 'aw', 'er', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'l', 'm', 'n', 'ng', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'sil'])
    print(f"length of cmu_dict: {len(awni_dct39)}")
    error_phones = []
    
    # looping over every phone list in the word_to_phoneme mapping
    for phones in word_to_phoneme.values():
        # looping over every phone in the specific list
        for phone in phones:
            if phone not in awni_dct39:
                error_phones.append(phone)
    error_phones = set(error_phones)
    print(f"there were {len(error_phones)} extra phones: {error_phones} ")



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

    path = os.path.join(args.output_directory, "LibriSpeech")   

    print("Converting files from flac to wave...")
    convert_to_wav(path)
    #save_lexicon_to_dict()
    for dataset, dirs in SETS.items():
        for d in dirs:
            print("Preprocessing {}".format(d))
            prefix = os.path.join(path, d)
            build_json(prefix, args.use_phonemes)
