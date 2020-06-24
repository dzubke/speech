# standard libraries
import argparse
import audioop
import glob
import os
import random
import subprocess
from tempfile import NamedTemporaryFile
from typing import Tuple
# third-party libraries
import librosa
import numpy as np
# project libraries
from speech.utils.wave import array_from_wave, array_to_wave, wav_duration
from speech.utils.convert import pcm2float, float2pcm
from speech.utils.data_structs import AugmentRange


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

    
    if augment_name == "synthetic_gaussian_noise_inject":
        snr_level = float(args)
        return synthetic_gaussian_noise_inject(audio_data, snr_range=(snr_level, snr_level))
    else:
        raise ValueError("augment_name doesn't match any augmentations")



# Speed_vol_perturb and augment_audio_with_sox code has been taken from 
# Sean Naren's Deepspeech implementation at:
# https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/data_loader.py

def tempo_gain_pitch_perturb(audio_path:str, sample_rate:int=16000, 
                            tempo_range:AugmentRange=(0.85, 1.15),
                            gain_range:AugmentRange=(-6.0, 8.0),
                            pitch_range:AugmentRange=(-400, 400), 
                            logger=None)->Tuple[np.ndarray, int]:
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns:
        tuple(np.ndarray, int) - the augmente audio data and the sample_rate
    """
    use_log = (logger is not None)
    if use_log: logger.info(f"tempo_gain_pitch_perturb: audio_file: {audio_path}")
    
    tempo_value = np.random.uniform(*tempo_range)
    if use_log: logger.info(f"tempo_gain_pitch_perturb: tempo_value: {tempo_value}")
    
    gain_value = np.random.uniform(*gain_range)
    if use_log: logger.info(f"tempo_gain_pitch_perturb: gain_value: {gain_value}")

    pitch_value = np.random.uniform(*pitch_range)
    if use_log: logger.info(f"tempo_gain_pitch_perturb: pitch_value: {pitch_value}")

    try:    
        audio_data, samp_rate = augment_audio_with_sox(audio_path, sample_rate, tempo_value, 
                                                        gain_value, pitch_value, logger=logger)
    except RuntimeError as rterr:
        if use_log: logger.error(f"tempo_gain_pitch_perturb: RuntimeError: {rterr}")
        audio_data, samp_rate = array_from_wave(audio_path)
        
    return audio_data, samp_rate 


def augment_audio_with_sox(path:str, sample_rate:int, tempo:float, gain:float, 
                            pitch:float, logger=None)->Tuple[np.ndarray,int]:
    """
    Changes tempo, gain (volume), and pitch of the recording with sox and loads it.
    """
    use_log = (logger is not None)
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_cmd = ['sox', '-V3',                # verbosity level = 3
                    path,                       # file to augment
                    '-r', f'{sample_rate}',     # sample rate
                    '-c', '1',                  # single-channel audio
                    '-b', '16',                 # bitrate = 16
                    '-e', 'si',                 # encoding = signed-integer
                    augmented_filename,         # output temp-filename
                    'tempo', f'{tempo:.3f}',    # augment tempo
                    'gain', f'{gain:.3f}',      # augment gain (in db)
                    'pitch', f'{pitch:.0f}']    # augment pitch (in hundredths of semi-tone)
        sox_result = subprocess.run(sox_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
        
        if use_log: 
            logger.info(f"aug_audio_sox: tmpfile exists: {os.path.exists(augmented_filename)}")
            logger.info(f"aug_audio_sox: sox stdout: {sox_result.stdout.decode('utf-8')}")
            stderr_message = sox_result.stderr.decode('utf-8')
            if 'FAIL' in stderr_message:
                logger.error(f"aug_audio_sox: sox stderr: {stderr_message}")
            else:
                logger.info(f"aug_audio_sox: sox stderr: {stderr_message}")
  
        
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
            noise_dst = audio_with_sox(noise_path, sample_rate, noise_start, noise_end, logger)
        except FileNotFoundError as fnf_err:
            if use_log: logger.error(f"noise_inject: FileNotFoundError: {fnf_err}")
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


def audio_with_sox(path:str, sample_rate:int, start_time:float, end_time:float, logger=None)\
                                                                                    ->np.ndarray:
    """
    crop and resample the recording with sox and loads it.
    If the output file cannot be found, an array of zeros of the desired length will be returned.
    """
    use_log = (logger is not None)
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_cmd = ['sox', '-V3',                # verbosity level=3
                    path,                       # noise filename
                    '-r', f'{sample_rate}',     # sample rate
                    '-c', '1',                  # output is single-channel audio
                    '-b', '16',                 # bitrate = 16
                    '-e', 'si',                 # encoding = signed-integer
                    tar_filename,               # output temp-filename
                     'trim', f'{start_time}', '='+f'{end_time}']    # trim to start and end time
        sox_result = subprocess.run(sox_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if use_log: 
            logger.info(f"noise_inj_sox: tmpfile exists: {os.path.exists(tar_filename)}")
            logger.info(f"noise_inj_sox: sox stdout: {sox_result.stdout.decode('utf-8')}")
            stderr_message = sox_result.stderr.decode('utf-8')
            if 'FAIL' in stderr_message:
                logger.error(f"aug_audio_sox: sox stderr: {stderr_message}")
            else:
                logger.info(f"aug_audio_sox: sox stderr: {stderr_message}")

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


# synthetic gaussian noise injection 
def synthetic_gaussian_noise_inject(audio_data: np.ndarray, snr_range:tuple=(10,30),
                                    logger=None):
    """
    Applies random noise to an audio sample scaled to a uniformly selected
    signal-to-noise ratio (snr) bounded by the snr_range

    Arguments:
        audio_data - np.ndarry: 1d array of audio amplitudes
        snr_range - tuple: range of values the signal-to-noise ratio (snr) can take on

    Note: Power = Amplitude^2 and here we are dealing with amplitudes = RMS
    """
    use_log = (logger is not None)
    snr_level = np.random.uniform(*snr_range)
    audio_rms = audioop.rms(audio_data, 2) 
    # 20 is in the exponent because we are dealing in amplitudes
    noise_rms = audio_rms / 10**(snr_level/20)
    gaussian_noise = np.random.normal(loc=0, scale=noise_rms, size=audio_data.size).astype('int16')
    augmented_data = audio_data + gaussian_noise
    
    if use_log: logger.info(f"syn_gaussian_noise: snr_level: {snr_level}")
    if use_log: logger.info(f"syn_gaussian_noise: audio_rms: {audio_rms}")
    if use_log: logger.info(f"syn_gaussian_noise: noise_rms: {noise_rms}")
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
    
    
