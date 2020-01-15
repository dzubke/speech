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
    "train" : ["train-other-500"],
    "dev" : ["dev-other"],
    "test" : ["test-other"]
    }

def load_transcripts(path, use_phonemes=True):
    pattern = os.path.join(path, "*/*/*.trans.txt")
    files = glob.glob(pattern)
    data = {}
    for f in files:
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
    for dataset, dirs in SETS.items():
        for d in dirs:
            print("Preprocessing {}".format(d))
            prefix = os.path.join(path, d)
            build_json(prefix, args.use_phonemes)
