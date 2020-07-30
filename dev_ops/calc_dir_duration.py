# standard libraries
import argparse
import glob
import os
# project libraries
from speech.utils.wave import wav_duration



def main(dir_path: str):

    pattern = "*.wav"
    dir_pattern = os.path.join(dir_path, pattern)
    audio_files = glob.glob(dir_pattern)

    total_duration = 0.0    # in seconds

    for audio_file in audio_files:
        #print("audio_file", audio_file)
        dur = wav_duration(audio_file)
        total_duration += dur
    
    print(f"total duration in directory: {round(total_duration, 3)} seconds")
    print(f"total duration in directory: {round(total_duration/60, 3)} minutes")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Calculates the total duration of all of the .wav files in a directory")
    parser.add_argument("--dir", type=str,
        help="Directory where the duration of the wav files will be calculated.")
    args = parser.parse_args()

    main(args.dir)