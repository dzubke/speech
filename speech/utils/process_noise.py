# standard library
import argparse
import os
import glob

# third-party libraries
from scipy.io.wavfile import write
import numpy as np

# project libraries
from speech.utils.wave import array_from_wave, wav_duration


def main(audio_dir:str, use_extend:str, use_resample:str) -> None: 
    """
        processes the background audio files mainly by duplicating audio files
        that are less than 60 seconds in length

    """

    if use_extend:
        target_duration = 60 # seconds
        extend_audio(audio_dir, target_duration)


def extend_audio(audio_dir:str, target_duration:int) -> None: 
    """
        stacks the audio files in audio_dur on themselves until they are each equal in
        length to the target_duration (in seconds)
        Arguments:
            audio_dir (str): directory of audio files
            target_duration (int): length in seconds the audio filles will be extended to
    """
    assert os.path.exists(audio_dir) == True, "audio directory does not exist"

    pattern = os.path.join(audio_dir, "*.wav")
    audio_files = glob.glob(pattern)
    
    for audio_fn in audio_files: 
        audio_duration = wav_duration(audio_fn)
        if audio_duration < target_duration:
            data, samp_rate = array_from_wave(audio_fn)
            # whole_dup as in whole_duplicate
            whole_dup, frac_dup = divmod(target_duration, audio_duration) 
            output_data = data
            #loop over whole_duplicates minus one because concatenating onto original
            for i in range(int(whole_dup)-1):
                output_data = np.concatenate((output_data, data), axis=0)
            # adding on the fractional section
            fraction_index = int(frac_dup*samp_rate)
            output_data = np.concatenate((output_data, data[:fraction_index]))

        file_name = os.path.basename(audio_fn)
        extended_name = file_name[:-4]+ "_extended.wav"
        extended_dir =  os.path.join(os.path.dirname(audio_fn), "extended")
        if not os.path.exists(extended_dir):
            os.mkdir(extended_dir)
        ext_audio_path = os.path.join(extended_dir, extended_name)

        write(ext_audio_path, samp_rate, output_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_dir", help="Directory that contains the background audio files.")
    parser.add_argument("--extend", action='store_true',
        help="Boolean flag that if present will call the extend_audio method ")
    args = parser.parse_args()

    main(args.audio_dir, args.extend)