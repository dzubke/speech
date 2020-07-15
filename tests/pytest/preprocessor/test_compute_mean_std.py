# standard libraries
import os
# third-party libraries
import numpy as np
import pytest
# project libraries
from speech.loader import compute_mean_std
from speech.utils.compat import get_main_dir_path
from speech.utils.io import read_pickle
from tests.pytest.utils import get_all_test_audio

# constants
WINDOW_SIZE = 32
STEP_SIZE = 16

def test_compute_mean_std_from_pickle():
    pytest_dir_path = os.path.join(get_main_dir_path(), "tests/pytest")
    # get_all_test_audio will pull everything from the test_audio directory
    # if more audio is added this will change the mean and std. 
    # thus, the same audio files must be imported which is why I am fixing the filenames below
    test_audio = [
        "Dustin-5-plane-noise.wav",
        "Speak-4ysq5X0Mvxaq1ArAntCWC2YkWHc2-1574725037.wav",
        "Speak-OVrsxD1n9Wbh0Hh6thej8FIBIOE2-1574725033.wav",
        "Librispeech-84-121123-0001.wav",
        "Speak-58cynYij95TbB9Nlz3TrKBbkg643-1574725017.wav",
        "Speak-out.wav"
    ]
    test_audio = list(map(lambda x: os.path.join(pytest_dir_path, "test_audio", x), test_audio))
   
    use_feature_normalize_options = [False, True]
    for use_feature_normalize in use_feature_normalize_options:
        print(f"=== feature_normalize: {use_feature_normalize} =====")
        mean, std = compute_mean_std(test_audio, 'log_spectrogram', 32, 16, use_feature_normalize)
        
        if use_feature_normalize:
            mean_pickle_path = os.path.join(pytest_dir_path, "test_pickle",
                "compute-mean-std-with-feature-normalize_log-spec_mean_2020-06-22.pickle")
            std_pickle_path = os.path.join(pytest_dir_path,"test_pickle",
                "compute-mean-std-with-feature-normalize_log-spec_std_2020-06-22.pickle")
        else:
            mean_pickle_path = os.path.join(pytest_dir_path, "test_pickle",
                                    "compute-mean-std_log-spec_mean_2020-06-22.pickle")
            std_pickle_path = os.path.join(pytest_dir_path, "test_pickle",
                                    "compute-mean-std_log-spec_std_2020-06-22.pickle")
        mean_reference = read_pickle(mean_pickle_path)
        std_reference = read_pickle(std_pickle_path)
        print(f"means shape: {mean.shape}, ref: {mean_reference.shape}")
        print(f"std shape: {std.shape}, ref: {std_reference.shape}")

        np.testing.assert_allclose(mean, mean_reference,  rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(std, std_reference,  rtol=1e-03, atol=1e-05)    



def test_single_input():
    """
    tests that a list with a single file will output a...
    """
    pass

def test_compute_mean_std_empty_input():
    """
    tests that an assertion is raised if an empyt list is sent as input
    """

    audio_files = list()
    preprocessor = 'log_spectrogram'
    use_feature_normalize_options = [False, True]
    for use_feature_normalize in use_feature_normalize_options:
        with pytest.raises(AssertionError) as execinfo:    
            mean, std = compute_mean_std(audio_files, preprocessor, WINDOW_SIZE, 
                                            STEP_SIZE, use_feature_normalize)
   
