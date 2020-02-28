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

PRONUNCIATION_LEXICON_PATH = "/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/examples/librispeech/librispeech-lexicon.txt"


def main(output_directory, use_phonemes):
    
    SETS = {
    "train" : ["train-clean-100"],
    "dev" : ["dev-clean"],
    "test" : ["test-clean "],
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
    transcripts, unknown_words = load_transcripts(path, use_phonemes)
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

    unk_words_fname = "libsp_"+basename+"_unk_words.txt"
    with open(unk_words_fname, 'w') as fid:
        for word in unknown_words:
            fid.write(word+'\n')

def load_transcripts(path, use_phonemes=True):
    pattern = os.path.join(path, "*/*/*.trans.txt")
    files = glob.glob(pattern)
    data = {}
    unknown_words=set()
    if use_phonemes: 
        word_phoneme_dict = data_helpers.lexicon_to_dict(PRONUNCIATION_LEXICON_PATH, corpus_name='librispeech')
        print(f"type of word_phoneme_dict: {type(word_phoneme_dict)}")
    for f in tqdm.tqdm(files):
        with open(f) as fid:
            lines = (l.strip().lower().split() for l in fid)
            if use_phonemes: 
                lines = ((l[0], transcript_to_phonemes(l[1:], word_phoneme_dict) ) for l in lines)
                unk_words = filter(lambda x: word_phoneme_dict[x] == "unk", word for l in lines for word in l[1:])
            else: 
                lines = ((l[0], " ".join(l[1:])) for l in lines)
            data.update(lines)
            unknown_words.update(set(unk_words))
    return data, unknown_words


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


def convert_to_wav(path):
    data_helpers.convert_full_set(path, "*/*/*/*.flac")


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