# standard libraries
import argparse
import audioop
import glob
import logging
from logging import Logger
import os
import random
import subprocess
import shutil
from tempfile import NamedTemporaryFile
from typing import Tuple
# third-party libraries
import librosa
import numpy as np
import scipy.stats      # need to include "stats" to aviod name-conflict
import yaml
# project libraries
from speech.utils.io import read_data_json
from speech.utils.data_structs import AugmentRange
from speech.utils.wave import array_from_wave, array_to_wave, wav_duration


def main(config:dict):
    data_cfg = config.get('data')
    log_cfg = config.get('logger')
    preproc_cfg = config.get('preproc')

    # create logger
    logger = logging.getLogger("sig_aug")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_cfg["log_file"])
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"config:\n{config}")

    dataset = read_data_json(data_cfg['data_set'])
    audio_list = [example['audio'] for example in dataset]
    audio_subset  = random.sample(audio_list, data_cfg['num_examples'])

    for audio_path in audio_subset: 
        aug_audio_data, samp_rate = apply_augmentation(audio_path, preproc_cfg, logger)
        
        os.makedirs(os.path.join(data_cfg['save_dir'], "aug"), exist_ok=True)
        os.makedirs(os.path.join(data_cfg['save_dir'], "org"), exist_ok=True)
        # save the augmented file
        basename = os.path.basename(audio_path)
        save_path = os.path.join(data_cfg['save_dir'], "aug", basename)
        array_to_wave(save_path, aug_audio_data, samp_rate)
        # copy the original audio file for comparison
        save_org_path = os.path.join(data_cfg['save_dir'], "org", basename)
        shutil.copyfile(audio_path, save_org_path)
        if preproc_cfg['play_audio']:
            print(f"sample rate: {sr}")
            print(f"Saved to: {save_path}")
            print("Playing original audio...")
            os_play(audio_path)
            print("Playing augmented audio...")
            os_play(save_path)

def os_play(play_file:str):
    play_str = f"play {play_file}"
    os.system(play_str)

def apply_augmentation(audio_path:str, preproc_cfg:dict, logger:Logger)\
                                                ->Tuple[np.ndarray, np.ndarray]:

    logger.info(f"audio_path: {audio_path}")
    if preproc_cfg['tempo_gain_pitch_perturb']:
        aug_data, samp_rate = tempo_gain_pitch_perturb(audio_path,
                                            tempo_range = preproc_cfg['tempo_range'],
                                            gain_range = preproc_cfg['gain_range'],
                                            pitch_range = preproc_cfg['pitch_range'],
                                            augment_from_normal = preproc_cfg['augment_from_normal'],
                                            logger= logger)
    else: 
        aug_data, samp_rate = array_from_wave(audio_path)

    if preproc_cfg['synthetic_gaussian_noise']:
        aug_data = synthetic_gaussian_noise_inject(aug_data, preproc_cfg['signal_to_noise_range_db'],
                                                        preproc_cfg['augment_from_normal'], logger=logger)
    if preproc_cfg['inject_noise']: 
        add_noise = np.random.binomial(1, preproc_cfg['noise_prob'])
        if add_noise:
            logger.info("noise injected")
            aug_data =  inject_noise(aug_data, samp_rate,  
                                        preproc_cfg['noise_directory'], 
                                        preproc_cfg['noise_levels'], 
                                        preproc_cfg['augment_from_normal'],
                                        logger) 
        else:
            logger.info("noise not injected")

    return aug_data, samp_rate



# Speed_vol_perturb and augment_audio_with_sox code has been taken from 
# Sean Naren's Deepspeech implementation at:
# https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/data_loader.py

def tempo_gain_pitch_perturb(audio_path:str, sample_rate:int=16000, 
                            tempo_range:AugmentRange=(0.85, 1.15),
                            gain_range:AugmentRange=(-6.0, 8.0),
                            pitch_range:AugmentRange=(-400, 400),
                            augment_from_normal:bool=False,
                            logger=None)->Tuple[np.ndarray, int]:
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Arguments:
        augment_from_normal - bool: if true, the augmentation values will be drawn from normal dist
    Returns:
        tuple(np.ndarray, int) - the augmente audio data and the sample_rate
    """
    use_log = (logger is not None)
    
    if augment_from_normal:
        tempo_center = np.mean(tempo_range)
        tempo_value = get_value_from_truncnorm(tempo_center, tempo_range, bounds=tempo_range)
        gain_center = np.mean(gain_range)
        gain_value = get_value_from_truncnorm(gain_center, gain_range, bounds=gain_range)
        pitch_center = np.mean(pitch_range)
        pitch_value = get_value_from_truncnorm(pitch_center, pitch_range, bounds=pitch_range)
    else:
        tempo_value = np.random.uniform(*tempo_range)
        gain_value = np.random.uniform(*gain_range)
        pitch_value = np.random.uniform(*pitch_range)

    if use_log: logger.info(f"tempo_gain_pitch_perturb: audio_file: {audio_path}")
    if use_log: logger.info(f"tempo_gain_pitch_perturb: tempo_value: {tempo_value}")
    if use_log: logger.info(f"tempo_gain_pitch_perturb: gain_value: {gain_value}")
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
                    '-t', 'wav',                # the output file is wav type
                    augmented_filename,         # output temp-filename
                    'tempo', f'{tempo:.3f}',    # augment tempo
                    'gain', f'{gain:.3f}',      # augment gain (in db)
                    'pitch', f'{pitch:.0f}']    # augment pitch (in hundredths of semi-tone)
        sox_result = subprocess.run(sox_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
        
        if use_log: 
            logger.info(f"sox_pertrub: aug_audio_sox: tmpfile exists: {os.path.exists(augmented_filename)}")
            logger.info(f"sox_pertrub: aug_audio_sox: sox stdout: {sox_result.stdout.decode('utf-8')}")
            stderr_message = sox_result.stderr.decode('utf-8')
            if 'FAIL' in stderr_message:
                logger.error(f"sox_pertrub: aug_audio_sox: sox stderr: {stderr_message}")
            else:
                logger.info(f"sox_pertrub: aug_audio_sox: sox stderr: {stderr_message}")
  
        
        data, samp_rate = array_from_wave(augmented_filename)
        return data, samp_rate


# Noise inject functions
def inject_noise(data, data_samp_rate, noise_dir, noise_levels=(0, 0.5), 
                    augment_from_normal:bool=False, logger=None):
    """
    injects noise from files in noise_dir into the input data. These
    methods require the noise files in noise_dir be resampled to 16kHz
    Arguments:
        augment_from_normal - bool: if true, augment value selected from normal distribution


    """
    use_log = (logger is not None)
    pattern = os.path.join(noise_dir, "*.wav")
    noise_files = glob.glob(pattern)    
    noise_path = np.random.choice(noise_files)
    if augment_from_normal:
        noise_level = get_value_from_truncnorm(center=0.0, value_range=noise_levels, bounds=noise_levels)
    else:
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

    if use_log: logger.info(f"noise_inj: noise duration (s): {noise_len}")
    if use_log: logger.info(f"noise_inj: data duration (s): {data_len}")

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
        
        noise_rms = np.sqrt(noise_dst.dot(noise_dst) / noise_dst.size)
        # avoid dividing by zero
        if noise_rms != 0:
            data_rms = np.sqrt(np.abs(data.dot(data)) / data.size)
            data += noise_level * noise_dst * data_rms / noise_rms

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
                    '-t', 'wav',                # the output file is wav type
                    tar_filename,               # output temp-filename
                     'trim', f'{start_time}', '='+f'{end_time}']    # trim to start and end time
        sox_result = subprocess.run(sox_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if use_log: 
            logger.info(f"noise_inj: sox: sox stdout: {sox_result.stdout.decode('utf-8')}")
            stderr_message = sox_result.stderr.decode('utf-8')
            if 'FAIL' in stderr_message:
                logger.error(f"noise_inj: sox: sox stderr: {stderr_message}")
                print(f"ERROR: noise_inj: sox: sox stderr: {stderr_message}")
            else:
                logger.info(f"noise_inj: sox: sox stderr: {stderr_message}")

        if os.path.exists(tar_filename):
            noise_data, samp_rate = array_from_wave(tar_filename)
        else:
            noise_len = round((end_time - start_time)/sample_rate)
            noise_data = np.zeros((noise_len,))
            logger.error(f"noise_inj: sox: tmp_file doesnt exist, using zeros of len {noise_len}")
            print(f"ERROR: noise_inj: sox: sox stderr: tmp_file doesnt exist, using zeros of len {noise_len}")
        
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
                                    augment_from_normal:bool=False, logger=None):
    """
    Applies random noise to an audio sample scaled to a uniformly selected
    signal-to-noise ratio (snr) bounded by the snr_range
    Arguments:
        audio_data - np.ndarry: 1d array of audio amplitudes
        snr_range - tuple: range of values the signal-to-noise ratio (snr) in dB
        augment_from_normal - bool: if true, augment values are chosen from normal distribution

    Note: Power = Amplitude^2 and here we are dealing with amplitudes = RMS
    """
    use_log = (logger is not None)
    if augment_from_normal:
        center = np.mean(snr_range)
        snr_level = get_value_from_truncnorm(center, value_range=snr_range, bounds=snr_range)
    else:
        snr_level = np.random.uniform(*snr_range)

    audio_rms = audioop.rms(audio_data, 2) 
    noise_rms = audio_rms / 10**(snr_level/20)    # 20 is in the exponent because we are dealing in amplitudes
    gaussian_noise = np.random.normal(loc=0, scale=noise_rms, size=audio_data.size).astype('int16')
    augmented_data = audio_data + gaussian_noise
    
    if use_log: logger.info(f"syn_gaussian_noise: snr_level: {snr_level}")
    if use_log: logger.info(f"syn_gaussian_noise: audio_rms: {audio_rms}")
    if use_log: logger.info(f"syn_gaussian_noise: noise_rms: {noise_rms}")
    assert augmented_data.dtype == "int16"
    
    return augmented_data


def get_value_from_truncnorm(center:int,
                             value_range:AugmentRange,
                             bounds:AugmentRange) -> float:
    """
    Returns a value from a normal distribution trunacated within a range of bounds. 
    """
    # ensures value_range and bounds are sorted from lowest to highest
    value_range.sort()
    bounds.sort()

    # setting range difference to be 3 standard devations from mean
    std_dev = abs(value_range[0] - value_range[1])/3
    # bound are compute relative to center/mean and std deviation
    lower_bound = (bounds[0] - center) / std_dev
    upper_bound = (bounds[1] - center) / std_dev

    value = scipy.stats.truncnorm.rvs(lower_bound, upper_bound, loc=center, scale=std_dev, size=None) 
    return float(value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment a file.")
    parser.add_argument("--config", help="Path to config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file) 

    main(config)
    
    
