# standard libraries
import argparse
import os
import json
import tqdm
import glob

from speech.utils import data_helpers
from speech.utils import wave

def load_phone_map():
    with open("phones.60-48-39.map", 'r') as fid:
        lines = (l.strip().split() for l in fid)
        lines = [l for l in lines if len(l) == 3]
    m60_48 = {l[0] : l[1] for l in lines}
    m48_39 = {l[1] : l[2] for l in lines}
    return m60_48, m48_39

def convert_to_wav(path):
    data_helpers.convert_full_set(path, "*.wav",
            new_ext='wv',
            use_avconv=False)

def load_transcripts(path):
    pattern = os.path.join(path, "*.PHN")
    m60_48, _ = load_phone_map()
    files = glob.glob(pattern)
    # Standard practic is to remove all "sa" sentences
    # for each speaker since they are the same for all.
    filt_sa = lambda x : os.path.basename(x)[:2] != "sa"
    files = filter(filt_sa, files)
    data = {}
    for f in files:
        with open(f) as fid:
            lines = (l.strip() for l in fid)
            phonemes = (l.split()[-1] for l in lines)
            phonemes = [m60_48[p] for p in phonemes if p in m60_48]
            data[f] = phonemes
    return data

def load_transcripts_list(path):
    pattern = os.path.join(path, "*.PHN")
    files = glob.glob(pattern)
    print(files)
    data = {}
    for f in files:
        with open(f) as fid:
            lst = [l.split() for l in fid]
            phonemes = [phn.lower() for phn in lst[0]]
            print(phonemes)
            data[f] = phonemes
    return data

def build_json(data, path, set_name):
    basename = set_name + os.path.extsep + "json"
    with open(os.path.join(path, basename), 'w') as fid:
        for k, t in tqdm.tqdm(data.items()):
            wave_file = os.path.splitext(k)[0] + os.path.extsep + 'wv'
            dur = wave.wav_duration(wave_file)
            datum = {'text' : t,
                     'duration' : dur,
                     'audio' : wave_file}
            json.dump(datum, fid)
            fid.write("\n")

if __name__ == "__main__":
    ## format of command is >>python test_preprocess.py <path_to_dataset> <json_name> --list_transcripts <True/False> 
    # where the optional --list_transcripts argument is whether the phoneme lable transcrips are in list form.    

    parser = argparse.ArgumentParser(
            description="Preprocess test dataset.")

    parser.add_argument("output_directory",
        help="Path where the dataset is saved.")

    parser.add_argument("json_filename",
        help="The name of the json file to be saved.")

    parser.add_argument("--list_transcripts",
        help="Boolean whether the transcrips are in list form.")

    args = parser.parse_args()
    print(f"args.output_directory: {args.output_directory}")

    path = os.path.abspath(args.output_directory)
    print(f"test dataset path: {path}")

    print("Converting files to standard wave format...")
    convert_to_wav(path)
    
    
    print("Preprocessing labels")
    if args.list_transcripts.lower() == 'true':
        test_data = load_transcripts_list(path)
    else:
        test_data = load_transcripts(path)

    print(f"train snippet: {list(test_data.items())[:2]}")

    print("Done loading transcripts")
    build_json(test_data, path, args.json_filename)
    