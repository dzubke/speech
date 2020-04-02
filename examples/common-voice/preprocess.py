# standard library
import csv
import os
import re
import argparse
import json
import subprocess
# third party libraries
import tqdm
# project libraries
from speech.utils import data_helpers, wave, convert


SETS = {
    "dev": "dev.tsv",
    "test": "test.tsv",
    "train":"train.tsv",
    "validated": "validated.tsv"
}

def main(cv_dir:str, lexicon_path:str, min_dur:float, max_dur:float, convert_wav:bool)->None:

    
    if lexicon_path !='':
        cv_dict = data_helpers.lexicon_to_dict(lexicon_path, corpus_name="common-voice")
        print(f"dict type:{type(cv_dict)}")
    else:
        cv_dict=None
    
    for set_name, label_name in SETS.items():
        label_fn = os.path.join(cv_dir, label_name)
        build_json(label_fn, cv_dict, min_dur, max_dur)

def build_json(label_fn:str, cv_dict:dict, min_dur:float, max_dur:float):

    data_dict = dict()
    basename = os.path.basename(label_fn)
    set_name = os.path.splitext(basename)[0]


    accents = ['us', 'canada']
    # open the file and select only entries with desired accents
    with open(label_fn) as fid: 
        reader = list(csv.reader(fid, delimiter='\t'))
        summary_stats(reader, set_name)
        print(f"Filtering files by accents: {accents}")

        for index, line in enumerate(reader):
            # first line is the header which equals:
            # ['client_id','path','sentence','up_votes','down_votes','age','gender','accent']
            if index == 0:
                continue
            else: 
                if line[7] in accents:
                    data_dict.update({index:{"audio":line[1], "trans": line[2]}})
    
    # filter the entries by the duration bounds and write file
    dirname = os.path.dirname(label_fn)
    json_path = os.path.join(dirname, set_name+".json")
    unknown_words = data_helpers.UnknownWords()
    with open(json_path, 'w') as fid:
        print("Writing files to label json")
        for i, audio_trans in tqdm.tqdm(data_dict.items()):
            mp3_file = os.path.join(dirname,"clips",audio_trans.get("audio"))
            if not os.path.exists(mp3_file):
                print(f"file {mp3_file} does not exists")
            else:
                base, mp3_ext = os.path.splitext(mp3_file)
                wav_file = base + os.path.extsep + "wav"
                if not os.path.exists(wav_file):
                    try:
                        convert.to_wave(mp3_file, wav_file)
                    except subprocess.CalledProcessError:
                        # if the file can't be converted, skip the file by continuing
                        print(f"Error converting file: {mp3_file}")
                        continue
                dur = wave.wav_duration(wav_file)
                if min_dur <= dur <= max_dur:
                    text = process_text(audio_trans.get("trans"), cv_dict, unknown_words, audio_trans.get("audio"))
                    # if transcript has an unknown word, skip it
                    if unknown_words.has_unknown: 
                        continue
                    datum = {'text' : text,
                            'duration' : dur,
                            'audio' : wav_file}
                    json.dump(datum, fid)
                    fid.write("\n")
    
    unk_path = os.path.join(dirname, set_name)
    print(f"saving unk-word-stats here: {unk_path}")
    unknown_words.process_save(unk_path)
    

def summary_stats(data:list, set_name:str):
    pass

def process_text(transcript:str, cv_dict:dict, unknown_words, audio_name:str)->list:
    # allows for alphanumeric characters, space, and apostrophe
    accepted_char = '[^A-Za-z0-9 \']+'
    # filters out unaccepted characters, lowers the case, & splits into list
    transcript = re.sub(accepted_char, '', transcript).lower()
    # check that all punctuation (minus apostrophe) has been removed 
    punct_noapost = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
    for p in punct_noapost:
        if p in transcript:
           raise ValueError(f"unwanted punctuation: {p} in transcript")
    #assert any([p in transcript for p in punct_noapost]), "unwanted punctuation in transcript"
    transcript = transcript.split()
    # if there is a pronunciation dict, convert to phonemes
    if cv_dict is not None:
        unknown_words.check_transcript(audio_name, transcript,  cv_dict)
        phonemes = []
        for word in transcript:
            if word is None:
                print(f"word is none")
            elif cv_dict.get(word) is None:
                print(f"dict entry is none for word: {word} and entry: {cv_dict.get(word)}")
            phonemes.extend(cv_dict.get(word, list()))
        return phonemes
    else:
        return transcript 





if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="creates a data json file")
    parser.add_argument("--cv-dir", type=str,
        help="directory where common voice .tsv files are located.")
    parser.add_argument("--lexicon-path", type=str, default='',
        help="path to pronunciation lexicon, if desired.")
    parser.add_argument("--min-duration", type=float,
        help="minimum audio duration in seconds")
    parser.add_argument("--max-duration", type=float,
        help="maximum audio duration in seconds")
    parser.add_argument("--convert-wav", action='store_true', default=False,
        help="directory where common voice .tsv files are located.")

    args = parser.parse_args()

    main(args.cv_dir, args.lexicon_path, args.min_duration, args.max_duration, args.convert_wav)