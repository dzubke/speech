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
    # "train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "dev-other", "test-clean", "test-other"  
    SETS = {
    "train" : ["train-clean-100", "train-clean-360", "train-other-500"],
    "dev" : ["dev-clean", "dev-other"],
    "test" : ["test-clean", "test-other"]
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
    unknown_set, unknown_dict = set(), dict()
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
                line_count+=counts[0]
                word_count+=counts[1]
                if len(unk_words_list) > 0: 
                    print(unk_words_list)
                    unknown_set.update(unk_words_list)
                    unknown_dict.update(unk_words_dict)
                    continue
                text = text.split()     #convert space-separated string to list of strings
                text = transcript_to_phonemes(text, word_phoneme_dict)
    
            datum = {'text' : text,
                     'duration' : dur,
                     'audio' : wave_file}
            json.dump(datum, fid)
            fid.write("\n")
    print(f"saving unk-word-stats here: {path}")
    data_helpers.process_unknown_words(path, unknown_set, unknown_dict, line_count, word_count)
    

def convert_to_wav(path):
    data_helpers.convert_full_set(path, "*/*/*/*.flac")


def load_transcripts(path):
    pattern = os.path.join(path, "*/*/*.trans.txt")
    files = glob.glob(pattern)
    data = {}
    for f in tqdm.tqdm(files):
        with open(f) as fid:
            lines = [l.strip().lower().split() for l in fid]
            lines = ((l[0], " ".join(l[1:])) for l in lines)
            data.update(lines)
    return data
    

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
