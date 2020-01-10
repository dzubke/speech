# standard libraries
import argparse
import os
import json
import tqdm
import glob

from speech.utils import data_helpers
from speech.utils import wave

def convert_to_wav(path):
    data_helpers.convert_full_set(path, "*.m4a",
            new_ext='wv',
            use_avconv=False)

def load_transcripts(path):
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
    parser = argparse.ArgumentParser(
            description="Preprocess test dataset.")

    parser.add_argument("output_directory",
        help="Path where the dataset is saved.")
    args = parser.parse_args()
    print(f"args.output_directory: {args.output_directory}")

    path = os.path.abspath(args.output_directory)
    print(f"test dataset path: {path}")

    print("Converting files to standard wave format...")
    convert_to_wav(path)

    print("Preprocessing labels")
    test_data = load_transcripts(path)

    print(f"train snippet: {list(test_data.items())[:2]}")

    print("Done loading transcripts")
    build_json(test_data, path, "speak_test")
