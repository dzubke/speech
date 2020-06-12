from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import soundfile

def array_from_wave(file_name):
    audio, samp_rate = soundfile.read(file_name, dtype='int16')
    return audio, samp_rate

def wav_duration(file_name):
    audio, samp_rate = soundfile.read(file_name, dtype='int16')
    nframes = audio.shape[0]
    duration = nframes / samp_rate
    return duration
 
def array_to_wave(filename:str, audio_data:np.ndarray, samp_rate:int):
    """
    Writes an array to wave in the in the signed int-16 subtype (PCM_16)
    """
    soundfile.write(filename, audio_data, samp_rate, subtype='PCM_16', format='WAV')