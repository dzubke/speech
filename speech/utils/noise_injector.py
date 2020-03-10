import os
import glob
from tempfile import NamedTemporaryFile

# third-party libraries
import numpy as np

# project libraries
from speech.utils.wave import wav_duration, array_from_wave

def inject_noise(data, data_samp_rate, noise_dir, noise_levels=(0, 0.5)):
    pattern = os.path.join(noise_dir, "*.wav")
    noise_files = glob.glob(pattern)
    noise_path = np.random.choice(noise_files)
    noise_level = np.random.uniform(*noise_levels)
    return inject_noise_sample(data, data_samp_rate, noise_path, noise_level)


def inject_noise_sample(data, sample_rate, noise_path, noise_level):
    noise_len = wav_duration(noise_path)
    data_len = len(data) / sample_rate
    noise_start = np.random.rand() * (noise_len - data_len)
    noise_end = noise_start + data_len
    noise_dst = audio_with_sox(noise_path, sample_rate, noise_start, noise_end)
    noise_dst = noise_dst.astype('float64')
    data = data.astype('float64')
    assert len(data) == len(noise_dst)
    noise_energy = np.sqrt(noise_dst.dot(noise_dst) / noise_dst.size)
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