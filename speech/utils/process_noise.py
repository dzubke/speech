# standard library
import argparse
import os
import glob

# third-party libraries
from scipy.io.wavfile import write
import numpy as np

# project libraries
from speech.utils.wave import array_from_wave, wav_duration
from speech.utils import convert


def main(audio_dir:str, use_extend:bool, use_resample:bool) -> None: 
    """
        processes the background audio files mainly by duplicating audio files
        that are less than 60 seconds in length
    """

    if use_extend:
        target_duration = 60 # seconds
        extend_audio(audio_dir, target_duration)
    
    if use_resample:
        target_samp_rate = 16000
        resample(audio_dir, target_samp_rate)


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


def resample(audio_dir:str, target_samp_rate:int)->None:
    """
    resamples all of the audio files in audio_dir to the target sample rate
    Arguments
        audio_dir (str): the audio directory whose files will be resampled
        target_samp_rate(int): the sample rate the files will be resampled to
    """

    assert os.path.exists(audio_dir) == True, "audio directory does not exist"
    out_dir = os.path.join(audio_dir, "resampled")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    extensions = ["*.wav", "*.mp3", "*.aiff", "*.flac"]
    audio_files = list()
    for ext in extensions:
        pattern = os.path.join(audio_dir, ext)
        audio_files.extend(glob.glob(pattern))
    
    for audio_fn in audio_files: 
        filename = os.path.splitext(os.path.basename(audio_fn))[0]
        wav_file = filename + os.path.extsep + "wav"
        out_path = os.path.join(out_dir, wav_file)
        convert.to_wave(audio_fn, out_path)
        # sox_params = "sox \"{}\" -r {} -c 1 -b 16 {}".format(audio_fn, target_samp_rate, out_path)
        # os.system(sox_params)

def resample_with_sox(path, sample_rate):
    """
    resample the recording with sox 
    """
    sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1".format(path, sample_rate,
                                                                                            tar_filename, start_time,
                                                                                            end_time)
    os.system(sox_params)
    noise_data, samp_rate = array_from_wave(tar_filename)
    return noise_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", help="Directory that contains the background audio files.")
    parser.add_argument("--extend", action='store_true', default=False,
        help="Boolean flag that if present will call the extend_audio method ")
    parser.add_argument("--resample", action='store_true', default=False,
        help="Boolean flag that if present will call the resample method ")
    args = parser.parse_args()

    main(args.audio_dir, args.extend, args.resample)