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

PRONUNCIATION_LEXICON_PATH = "librispeech-lexicon.txt"


def main(output_directory, use_phonemes):
    
    SETS = {
    "train" : ["train-clean-360", "train-other-500"],
    "dev" : ["dev-other"],
    "test" : ["test-other"],
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
    transcripts, unknown_words_set, unknown_words_dict = load_transcripts(path, use_phonemes)
    print(f"unknown words dict: {len(unknown_words_dict)}")
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

    process_unknown_words(path, unknown_words_set, unknown_words_dict)
    

def convert_to_wav(path):
    data_helpers.convert_full_set(path, "*/*/*/*.flac")


def load_transcripts(path, use_phonemes=True):
    pattern = os.path.join(path, "*/*/*.trans.txt")
    files = glob.glob(pattern)
    data = {}
    unknown_set=set()
    unknown_dict=dict()
    if use_phonemes: 
        word_phoneme_dict = data_helpers.lexicon_to_dict(PRONUNCIATION_LEXICON_PATH, corpus_name="librispeech")
        print(f"type of word_phoneme_dict: {type(word_phoneme_dict)}")
    for f in tqdm.tqdm(files):
        with open(f) as fid:
            # load transcript of file
            lines = (l.strip().lower().split() for l in fid) 
            if use_phonemes: 
                file_unk_list, file_unk_dict= check_unknown_words(lines, word_phoneme_dict)
                lines = ((l[0], transcript_to_phonemes(l[1:], word_phoneme_dict) ) for l in lines)
                unknown_set.update(file_unk_list)
                unknown_dict.update(file_unk_dict)
            else: 
                lines = ((l[0], " ".join(l[1:])) for l in lines)
                unk_words = []
            data.update(lines)
    return data, unknown_set, unknown_dict


def check_unknown_words(lines, word_phoneme_dict):
    unk_words_list = list()
    unk_words_dict = dict()
    for line in lines:
        line_name = line[0] 
        line_unk_list = [word for word in line[1:] if word_phoneme_dict[word] =="unk"]
        if line_unk_list:       #if not empty
            unk_words_list.extend(line_unk_list)
            unk_words_dict.update({line_name: len(line_unk_list)})

    return unk_words_list, unk_words_dict


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


def process_unknown_words(path, unknown_words_set, unknown_words_dict):
    """saves a json object of the dictionary with relevant statistics on the unknown words in corpus
    """

    stats_dict=dict()
    stats_dict.update({"unique_unknown_words": len(unknown_words_set)})
    stats_dict.update({"count_unknown_words": sum(unknown_words_dict.values())})
    stats_dict.update({"lines_unknown_words": len(unknown_words_dict)})
    stats_dict.update({"unknown_words_set": list(unknown_words_set)})
    stats_dict.update({"unknown_words_dict": unknown_words_dict})

    stats_dict_fname = "libsp_"+os.path.basename(path)+"_unk-words-stats.json"
    with open(stats_dict_fname, 'w') as fid:
        json.dump(stats_dict, fid)


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
