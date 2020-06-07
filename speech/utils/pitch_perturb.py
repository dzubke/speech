# standard libraries
import argparse
import os
import random
from tempfile import NamedTemporaryFile
# third-party libraries
import librosa
import numpy as np
# project libraries
from speech.utils.wave import array_from_wave, array_to_wave
from speech.utils.convert import pcm2float, float2pcm



def main(audio_path: str, out_path:str, quarter_steps:int):
    audio_data, sr = array_from_wave(audio_path)
    quarter_steps = int(quarter_steps)
    aug_audio_data = apply_pitch_perturb(audio_data, sr, lower_range=quarter_steps, upper_range=quarter_steps)

    with NamedTemporaryFile(suffix=".wav") as tmp_file:
        tmp_filename = tmp_file.name

        save_path = tmp_filename if out_path is None else out_path
        array_to_wave(save_path, aug_audio_data, sr)
        print(f"sample rate: {sr}")
        print(f"Saved to: {save_path}")
        os_play(audio_path)
        os_play(save_path)

def os_play(play_file:str):
    play_str = f"play {play_file}"
    os.system(play_str)

def apply_pitch_perturb(audio_data:np.ndarray, samp_rate=16000, lower_range=-8, upper_range=8):
    """
    Adjusts the pitch of the input audio_data by selecting a random value between
    the lower and upper ranges and adjusting the pitch based on the chosen number of
    quarter steps
    Arguments:
        audio_data - np.ndarray: array of audio amplitudes
        samp_rate - int: sample rate of input audio
        lower_range - int: minimum number of quarter steps to drop the pitch. 
        upper_range - int: maximum number of quarter steps to drop the pitch. 
    Returns: 
        augment_data - np.ndarrray: array of audio amplitudes with raised pitch
    """
    assert audio_data.size >= 2, "input data must be 2 or more long"
    random_steps = random.randint(lower_range, upper_range)
    audio_data = pcm2float(audio_data)
    augment_data = librosa.effects.pitch_shift(audio_data, samp_rate, n_steps=random_steps, bins_per_octave=24)
    augment_data = float2pcm(augment_data)
    return augment_data





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adjust the pitch of a file and play new file")
    parser.add_argument("--audio-path",
        help="Path to input audio file.")
    parser.add_argument("--out-path", default=None,
        help="Path the augmented file will be saved to.")
    parser.add_argument("--quarter-steps",
        help="Number of half stesp to augment the file.")
    ARGS = parser.parse_args()

    main(ARGS.audio_path, ARGS.out_path, ARGS.quarter_steps)
    
    