import os
import glob
from tempfile import NamedTemporaryFile

# third-party libraries
import numpy as np

# project libraries
from speech.utils.wave import wav_duration, array_from_wave

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
    if use_log: logger.info(f"noise_inj: noise_path: {noise_path}")
    noise_level = np.random.uniform(*noise_levels)
    if use_log: logger.info(f"noise_inj: noise_level: {noise_level}")
    return inject_noise_sample(data, data_samp_rate, noise_path, noise_level, logger)


def inject_noise_sample(data, sample_rate:int, noise_path:str, noise_level:float, logger):
    """
    Takes in a numpy array (data) and adds a section of the audio in noise_path
    to the numpy array in proprotion on the value in noise_level
    """
    use_log = (logger is not None)
    noise_len = wav_duration(noise_path)
    if use_log: logger.info(f"noise_inj: noise_len: {noise_len}")
    data_len = len(data) / sample_rate
    if use_log: logger.info(f"noise_inj: data_len: {data_len}")
    if data_len > noise_len: # if the noise_file len is too small, skip it
        return data
    else:
        noise_start = np.random.rand() * (noise_len - data_len) 
        if use_log: logger.info(f"noise_inj: noise_start: {noise_start}")
        noise_end = noise_start + data_len
        if use_log: logger.info(f"noise_inj: noise_end: {noise_end}")
        noise_dst = audio_with_sox(noise_path, sample_rate, noise_start, noise_end)
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
        return data.astype('int16')


def audio_with_sox(path, sample_rate, start_time, end_time):
    """
    crop and resample the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1".format(path, sample_rate,
                                                                                               tar_filename, start_time,
                                                                                               end_time)
        os.system(sox_params)
        noise_data, samp_rate = array_from_wave(tar_filename)
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
