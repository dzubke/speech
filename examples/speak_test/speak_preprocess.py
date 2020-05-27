# standard libraries
import argparse
import csv
import json
import os
# project libraries
from speech.utils.convert import to_wave
from speech.utils.wave import wav_duration

def main(label_csv:str, audio_dir:str, json_path:str)->None:
    """
    Reads the label_csv and writes the labels, audio_path, and duration
    to the path in json_path. 
    """

    with open(json_path, 'w') as fid:
        with open(label_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            
            for line in reader:
                assert len(line)==2, f"row with {line[0]} has more than 2 elements"
                src_ext = "m4a"
                src_filename = line[0] + os.path.extsep + src_ext
                src_audio_path = os.path.join(audio_dir, src_filename)

                dst_ext = "wav"
                dst_filename = line[0] + os.path.extsep + dst_ext
                dst_audio_path = os.path.join(audio_dir, dst_filename)

                to_wave(src_audio_path, dst_audio_path)

                labels = line[1]
                labels = process_labels(labels)
                
                duration  = wav_duration(dst_audio_path)

                datum = {'text' : labels,
                        'duration' : duration,
                        'audio' : dst_audio_path}
                
                json.dump(datum, fid)
                fid.write("\n")


def process_labels(labels:str):
    """
    Takes a string of phonemes as input and outputs a list of lowercase phonemes
    """
    return labels.lower().split()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Creates a data.json from the speak test set')
    parser.add_argument("--label-csv", type=str, required=True,
        help="csv that contains the phone labels and audio filenames.")
    parser.add_argument("--audio-dir", type=str, required=True,
        help="the directory where the audio files are located.")
    parser.add_argument("--json-path", type=str, required=True,
        help="the file path where the json will be saved.")
    args = parser.parse_args()

    main(args.label_csv, args.audio_dir, args.json_path)