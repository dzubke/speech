# standard libraries
import glob
import os
import platform
# project libraries
from speech.utils.wave import array_from_wave
from speech.utils.signal_augment import inject_noise_sample
from speech.utils.compat import get_main_dir_path

def check_length(audio_path:str, noise_path:str, noise_level:float=0.5):
    audio_data, samp_rate = array_from_wave(audio_path)
    audio_noise = inject_noise_sample(audio_data, samp_rate, noise_path, 
                    noise_level=noise_level, logger=None)

def get_all_test_audio():
    
    system_main_dir = get_main_dir_path() 
    common_path = "tests/pytest/test_audio"
    test_audio_dir = os.path.join(system_main_dir, common_path)
    pattern = "*"
    return glob.glob(os.path.join(test_audio_dir, pattern))
