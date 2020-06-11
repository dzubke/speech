# standard libraries
import glob
import os
# project libraries
from speech.utils.wave import array_from_wave
from speech.utils.signal_augment import inject_noise_sample

def check_length(audio_path:str, noise_path:str, noise_level:float=0.5):
    audio_data, samp_rate = array_from_wave(audio_path)
    audio_noise = inject_noise_sample(audio_data, samp_rate, noise_path, 
                    noise_level=noise_level, logger=None)

def get_all_test_audio():
    test_audio_dir = "/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/tests/pytest/test_audio"
    pattern = "*"
    return glob.glob(os.path.join(test_audio_dir, pattern))