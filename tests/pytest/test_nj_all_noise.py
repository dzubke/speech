# standard libraries
import json
# third party libraries
import pytest
import tqdm
# project libraries
from speech import dataset_info
from speech.utils.wave import array_from_wave
from speech.utils.signal_augment import audio_with_sox
import utils


def test_main():
    
    # runs all the noise files against a set of audio files
    check_all_noise()
    
    # tests a set of noise and audio files against a range of noise_levels
    check_noise_level()

    # tests audio_with_sox utility that creates a range of segments for all noise files 
    #check_audio_with_sox()


def check_all_noise():
    noise_dataset = dataset_info.NoiseDataset()
    noise_files = noise_dataset.files_from_pattern()
    audio_17s ="/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/19/198/19-198-0034.wav"
    audio_2s = "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/19/198/19-198-0000.wav"
    test_audio = [audio_2s, audio_17s]
    print(f"\n Test All Noise: testing {len(noise_files)} noise files")
    for audio_file in test_audio:
        for noise_file in noise_files:
            try:
                utils.check_length(audio_file, noise_file)
            except AssertionError:
                raise AssertionError(f"audio: {audio_file} and noise: {noise_file}")
            except FileNotFoundError:
                raise FileNotFoundError(f"audio: {audio_file} and noise: {noise_file}")
            #except:
                #raise Exception(f"audio: {audio_file}, noise: {noise_file}")
            
def check_noise_level():
    """
    this test aims to test noise inject using a variety of noise levels
    across a selection of noise files and test audio files
    """
    noise_files = [
        "/home/dzubke/awni_speech/data/background_noise/100263_43834-lq.wav",
        "/home/dzubke/awni_speech/data/background_noise/101281_1148115-lq.wav",
        "/home/dzubke/awni_speech/data/background_noise/102547_1163166-lq.wav",
        "/home/dzubke/awni_speech/data/background_noise/elaborate_thunder-Mike_Koenig-1877244752.wav",
        "/home/dzubke/awni_speech/data/background_noise/violet_noise_2.wav",
        "/home/dzubke/awni_speech/data/background_noise/115418_8043-lq.wav"
   ]
    # first test audio is 17 s, second is 2 s, third is from separate dataset
    test_audio = [
        "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/19/198/19-198-0034.wav",
        "/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/19/198/19-198-0000.wav",
        "/home/dzubke/awni_speech/data/tedlium/TEDLIUM_release-3/data/converted/wav/EmmanuelJal_2009G_104.wav.wav"
    ]
    # making a list of noise_levels form 0 to 1.15 in increments of 0.5
    noise_levels = [x/100 for x in range(0,120, 5)]     

    print(f"\n Noise Level Test: testing {len(noise_files)} noise files")
    for audio_file in test_audio:
        for noise_file in noise_files:
            for noise_level in noise_levels:
                try:
                    utils.check_length(audio_file, noise_file, noise_level=noise_level)
                except AssertionError:
                    raise AssertionError(f"audio:{audio_file}, noise:{noise_file}, noise_level:{noise_level}")
                except FileNotFoundError:
                    raise FileNotFoundError(f"audio:{audio_file}, noise:{noise_file}, noise_level:{noise_level}")
                except:
                    raise Exception(f"audio:{audio_file}, noise:{noise_file}, noise_level:{noise_level}")

def check_audio_with_sox():
    """
    this test aims to find files where audio_with_sox raises a 
    FileNotFoundError by running audio_with_sox over the entire
    noise file using different window sizes defined in data_lens 
    """
    noise_dataset = dataset_info.NoiseDataset()
    noise_files = noise_dataset.files_from_pattern()
    data_lens = [0.5, 5, 50] # in secs
    step_size = 0.05
    print(f"\n Test Full Noise File: testing {len(noise_files)} noise files...")
    file_count = 0
    for noise_file in noise_files:
        print(f"Processing file {file_count}: {noise_file}")
        file_count += 1
        audio, samp_rate = array_from_wave(noise_file)
        noise_len = audio.shape[0] / samp_rate
        for data_len in data_lens:
            start_end_tups = calc_start_end(noise_len, data_len, step_size)
            for noise_start, noise_end in start_end_tups:
                try:
                    noise_dst = audio_with_sox(noise_file, samp_rate, noise_start, noise_end)
                except AssertionError:
                    raise AssertionError(f"noise:{noise_file}, data_len: {data_len}")
                except FileNotFoundError:
                    raise FileNotFoundError(f"noise:{noise_file}, data_len: {data_len}")
                except:
                    raise Exception(f"noise:{noise_file}, data_len: {data_len}")


def calc_start_end(noise_len:float, data_len:float, step_size:float)->list:
    """
    returns a list of tuples of the start and end times
    that specify the data_len window moving across noise_len
    with a step specified by step_size
    """
    start_end = list()
    noise_start = 0.0
    noise_end = noise_start + data_len
    while (noise_end < noise_len):
        start_end.append( (noise_start, noise_end))
        noise_start += step_size
        noise_end = noise_start + data_len
    return start_end
