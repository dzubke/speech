# standard libraries
import argparse
import audioop
import glob
import os
import random
from tempfile import NamedTemporaryFile
# third-party libraries
import librosa
import numpy as np
# project libraries
from speech.utils.wave import array_from_wave, array_to_wave, wav_duration
from speech.utils.convert import pcm2float, float2pcm



def main(audio_path: str, out_path:str, augment_name:str, ARGS):
    audio_data, sr = array_from_wave(audio_path)

    aug_audio_data = apply_augmentation(audio_data, sr, augment_name, ARGS)

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

def apply_augmentation(audio_data:np.ndarray, sr:int, augment_name:str, args):

    
    if augment_name == "pitch_perturb":
        pitch_level = int(args)
        return apply_pitch_perturb(audio_data, sr, pitch_range=(pitch_level, pitch_level))
    elif augment_name == "synthetic_gaussian_noise_inject":
        snr_level = float(args)
        return synthetic_gaussian_noise_inject(audio_data, snr_range=(snr_level, snr_level))
    else:
        raise ValueError("augment_name doesn't match any augmentations")


    

def apply_pitch_perturb(audio_data:np.ndarray, samp_rate:int=16000, pitch_range:tuple=(-8,8)):
    """
    Adjusts the pitch of the input audio_data by selecting a random value between the lower
    and upper ranges and adjusting the pitch based on the chosen number of quarter steps
    Arguments:
        audio_data - np.ndarray: array of audio amplitudes
        samp_rate - int: sample rate of input audio
        pitch_range - tuple: min and max number of quarter steps to drop the pitch. 
    Returns: 
        augment_data - np.ndarrray: array of audio amplitudes with raised pitch
    """
    assert audio_data.size >= 2, "input data must be 2 or more long"
    random_steps = random.randint(*pitch_range)
    audio_data = pcm2float(audio_data)
    augment_data = librosa.effects.pitch_shift(audio_data, samp_rate, n_steps=random_steps, bins_per_octave=24)
    augment_data = float2pcm(augment_data)
    return augment_data



# Speed_vol_perturb and augment_audio_with_sox code has been taken from 
# Sean Naren's Deepspeech implementation at:
# https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/data_loader.py

def speed_vol_perturb(path, sample_rate=16000, tempo_range=(0.85, 1.15),
                                  gain_range=(-6, 8))->tuple:
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio, samp_rate = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value)
    return audio, samp_rate 

def augment_audio_with_sox(path, sample_rate, tempo, gain)->tuple:
    """
    Changes speed (tempo) and volume (gain) of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                                      augmented_filename,
                                                                                      " ".join(sox_augment_params))
        os.system(sox_params)
        data, samp_rate = array_from_wave(augmented_filename)
        return data, samp_rate


# Noise inject functions

def inject_noise(data, data_samp_rate, noise_dir, logger, noise_levels=(0, 0.5)):
    """
    injects noise from files in noise_dir into the input data. These
    methods require the noise files in noise_dir be resampled to 16kHz
    with process_noise.py in speech.utils.
    """
    use_log = (logger is not None)
    pattern = os.path.join(noise_dir, "*.wav")
    noise_files = glob.glob(pattern)    
    noise_path = np.random.choice(noise_files)
    noise_level = np.random.uniform(*noise_levels)

    if use_log: logger.info(f"noise_inj: noise_path: {noise_path}")
    if use_log: logger.info(f"noise_inj: noise_level: {noise_level}")

    return inject_noise_sample(data, data_samp_rate, noise_path, noise_level, logger)


def inject_noise_sample(data, sample_rate:int, noise_path:str, noise_level:float, logger):
    """
    Takes in a numpy array (data) and adds a section of the audio in noise_path
    to the numpy array in proprotion on the value in noise_level
    """
    use_log = (logger is not None)
    noise_len = wav_duration(noise_path)
    data_len = len(data) / sample_rate

    if use_log: logger.info(f"noise_inj: noise_len: {noise_len}")
    if use_log: logger.info(f"noise_inj: data_len: {data_len}")

    if data_len > noise_len: # if the noise_file len is too small, skip it
        return data
    else:
        noise_start = np.random.rand() * (noise_len - data_len) 
        noise_end = noise_start + data_len
        try:
            noise_dst = audio_with_sox(noise_path, sample_rate, noise_start, noise_end)
        except FileNotFoundError:
            if use_log: logger.info(f"file not found error in: audio_with_sox")
            return data

        noise_dst = same_size(data, noise_dst)
        # convert to float to avoid value integer overflow in .dot() operation
        noise_dst = noise_dst.astype('float64')
        data = data.astype('float64')
        assert len(data) == len(noise_dst), f"data len: {len(data)}, noise len: {len(noise_dst)}, data size: {data.size}, noise size: {noise_dst.size}, noise_path: {noise_path}"
        
        noise_energy = np.sqrt(noise_dst.dot(noise_dst) / noise_dst.size)
        # avoid dividing by zero
        if noise_energy != 0:
            data_energy = np.sqrt(np.abs(data.dot(data)) / data.size)
            data += noise_level * noise_dst * data_energy / noise_energy

        if use_log: logger.info(f"noise_inj: noise_start: {noise_start}")
        if use_log: logger.info(f"noise_inj: noise_end: {noise_end}")

        return data.astype('int16')


def audio_with_sox(path:str, sample_rate:int, start_time:float, end_time:float)->np.ndarray:
    """
    crop and resample the recording with sox and loads it.
    If the output file cannot be found, an array of zeros of the desired length will be returned.
    """
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1".format(path, sample_rate,
                                                                                               tar_filename, start_time,
                                                                                               end_time)
        os.system(sox_params)
        
        if os.path.exists(tar_filename):
            noise_data, samp_rate = array_from_wave(tar_filename)
        else:
            noise_len = round((end_time - start_time)/sample_rate)
            noise_data = np.zeros((noise_len,))
        
        assert isinstance(noise_data, np.ndarray), "not numpy array returned"
        return noise_data

def same_size(data:np.ndarray, noise_dst:np.ndarray) -> np.ndarray:
    """
    this function adjusts the size of noise_dist if it is smaller or bigger than the size of data
    """

    if data.size == noise_dst.size:
        return noise_dst
    elif data.size < noise_dst.size:
        size_diff = noise_dst.size - data.size
        return noise_dst[:-size_diff]
    elif data.size > noise_dst.size:
        size_diff = data.size - noise_dst.size
        zero_diff = np.zeros((size_diff))
        return np.concatenate((noise_dst, zero_diff), axis=0)


# random noise injection 

def signal_noise_inject(audio_data: np.ndarray, snr_range:tuple=(10,30)):
    """
    Applies random noise to an audio sample scaled to a uniformly selected
    signal-to-noise ratio (snr) bounded by the snr_range

    Arguments:
        audio_data - np.ndarry: 1d array of audio amplitudes
        snr_range - tuple: range of values the signal-to-noise ratio (snr) can take on
    """
    audio_data = audio_data.astype('float64')
    std_norm_noise = np.random.normal(loc=0, scale=1, size=audio_data.size).astype('float64')
    snr_level = np.random.uniform(*snr_range)
    audio_power = audio_data.dot(audio_data) / audio_data.size
    noise_power = std_norm_noise.dot(std_norm_noise) / std_norm_noise.size
    power_ratio = int(audio_power/noise_power)
    noise_adj_factor = power_ratio / 10**(snr_level/10)
    #print("v1 sqrt noise_adj", np.sqrt(noise_adj_factor))
    return (audio_data + std_norm_noise * np.sqrt(noise_adj_factor)).astype('int16')

def synthetic_gaussian_noise_inject(audio_data: np.ndarray, snr_range:tuple=(10,30)):
    """
    Applies random noise to an audio sample scaled to a uniformly selected
    signal-to-noise ratio (snr) bounded by the snr_range

    Arguments:
        audio_data - np.ndarry: 1d array of audio amplitudes
        snr_range - tuple: range of values the signal-to-noise ratio (snr) can take on

    Note: Power = Amplitude^2 and here we are dealing with amplitudes = RMS
    """
    snr_level = np.random.uniform(*snr_range)
    audio_rms = audioop.rms(audio_data, 2) 
    # 20 is in the exponent because we are dealing in amplitudes
    noise_rms = audio_rms / 10**(snr_level/20)
    gaussian_noise = np.random.normal(loc=0, scale=noise_rms, size=audio_data.size).astype('int16')
    augmented_data = audio_data + gaussian_noise
    assert augmented_data.dtype == "int16"
    return augmented_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adjust the pitch of a file and play new file")
    parser.add_argument("--audio-path",
        help="Path to input audio file.")
    parser.add_argument("--augment-name",
        help="Name of augmentation applied.")
    parser.add_argument("--out-path", default=None,
        help="Path the augmented file will be saved to.")
    parser.add_argument("--quarter-steps",
        help="Number of half stesp to augment the file.")
    ARGS = parser.parse_args()

    main(ARGS.audio_path, ARGS.out_path, ARGS.augment_name, ARGS.quarter_steps)
    
    