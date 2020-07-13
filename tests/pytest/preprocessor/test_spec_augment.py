# standard libraries
from logging import Logger
import logging
# third party libraries
import numpy as np
import torch
import pytest
from _pytest.fixtures import SubRequest
# project libraries
from speech.loader import log_spectrogram_from_data
from speech.utils.convert import to_numpy
from speech.utils.feature_augment import apply_spec_augment, spec_augment
from speech.utils.wave import array_from_wave
from tests.pytest.utils import get_all_test_audio



def test_apply_spec_augment_call(logger:Logger=None):
    """
    Just tests if the apply_spec_augment can be called without errors
    Arguments:
        logger - Logger: can be taken as input to teset logger
    """
    audio_paths = get_all_test_audio()
    for audio_path in audio_paths:
        audio_data, samp_rate = array_from_wave(audio_path)
        features = log_spectrogram_from_data(audio_data, samp_rate, window_size=32, step_size=16)
        apply_spec_augment(features, logger)


def test_freq_masking(logger:Logger=None):
    """
    Checks that the number of frequency masks are less than the maximum number allowed. 
    Values of test_tuples are:
    ('time_warping_para', 'frequency_masking_para', 'time_masking_para'
    'frequency_mask_num',  'time_mask_num')
    """
    test_tuples =  [(0, 60, 0, 1, 0),   # 1 mask with max width of 60
                    (0, 30, 0, 2, 0),
                    (0, 20, 0, 3, 0)
    ]
    audio_paths = get_all_test_audio()
    number_of_tests = 10 # multiple tests as mask selection is random
    for _ in range(number_of_tests):
        for audio_path in audio_paths:    
            for param_tuple in test_tuples:
                audio_data, samp_rate = array_from_wave(audio_path)
                features = log_spectrogram_from_data(audio_data, samp_rate, window_size=32, step_size=16)
                features = torch.from_numpy(features.T)
                aug_features = spec_augment(features, *param_tuple)
                aug_features = to_numpy(aug_features)
                num_mask_rows = count_freq_mask(aug_features)

                freq_mask_size = param_tuple[1]
                num_freq_masks = param_tuple[3]
                max_freq_masks = freq_mask_size * num_freq_masks
                
                #print(f"number of masked rows: {num_mask_rows}, max_masked: {max_freq_masks}")
                assert  num_mask_rows<= max_freq_masks





def count_freq_mask(array:np.ndarray)->bool:
    """
    Counts the number of frequency masked rows
    Arguments:
        array - np.ndarray: 2d numpy array with dimension frequency x time
    """
    count_zero_rows = 0
    for row_index in range(array.shape[0]):
        if array[row_index, 0] == 0:
            if np.sum(array[row_index, :]) == 0:
                count_zero_rows += 1
    return count_zero_rows


def test_time_masking(logger:Logger=None):
    """
    Checks that the number of time masks are less than the maximum number allowed. 
    Values of test_tuples are:
    ('time_warping_para', 'frequency_masking_para', 'time_masking_para'
    'frequency_mask_num',  'time_mask_num')
    """
    test_tuples =  [(0, 0, 60, 0, 1),   # 1 mask with max width of 60
                    (0, 0, 30, 0, 2),
                    (0, 0, 20, 0, 3)
    ]
    audio_paths = get_all_test_audio()
    number_of_tests = 10 # multiple tests as mask selection is random
    for _ in range(number_of_tests):
        for audio_path in audio_paths:    
            for param_tuple in test_tuples:
                audio_data, samp_rate = array_from_wave(audio_path)
                features = log_spectrogram_from_data(audio_data, samp_rate, window_size=32, step_size=16)
                features = torch.from_numpy(features.T)
                aug_features = spec_augment(features, *param_tuple)
                aug_features = to_numpy(aug_features)
                num_mask_rows = count_time_mask(aug_features)

                time_mask_size = param_tuple[2]
                num_time_masks = param_tuple[4]
                max_time_masks = time_mask_size * num_time_masks
                
                #print(f"number of time masked rows: {num_mask_rows}, max_time_masked: {max_time_masks}")
                assert  num_mask_rows<= max_time_masks


def count_time_mask(array:np.ndarray)->bool:
    """
    Counts the number of time masked rows
    Arguments:
        array - np.ndarray: 2d numpy array with dimension frequency x time
    """
    count_zero_columns = 0
    for col_index in range(array.shape[1]):
        if array[0][col_index] == 0:
            if np.sum(array[:, col_index]) == 0:
                count_zero_columns += 1
    return count_zero_columns



def test_logger():
    """
    Runs all the tests with a logger to test if running with logger fails
    """
    logging.basicConfig(filename=None, filemode='w', level=10)
    logger = logging.getLogger("train_log")

    test_apply_spec_augment_call(logger)


def test_for_nan_values():
    """
    this test will try a variety of audio files and input parameters to generate nan values
    """
    logging.basicConfig(filename=None, filemode='w', level=10)
    logger = logging.getLogger("train_log")

    test_audio = get_nan_audio()
    params_list = get_nan_parameters()

    for audio_count, audio_path in enumerate(test_audio):
        for params_count, params_dict in enumerate(params_list):
            audio_data, samp_rate = array_from_wave(audio_path)
            features = log_spectrogram_from_data(audio_data, samp_rate, window_size=32, step_size=16)
            features = torch.from_numpy(features.T)
            features = spec_augment(features,
                                time_warping_para = params_dict["W"],
                                frequency_masking_para = params_dict["frequency_masking_para"],
                                time_masking_para = params_dict["time_masking_para"],
                                frequency_mask_num = len(params_dict["f"]),
                                time_mask_num = len(params_dict["t"]),
                                logger=logger,
                                fixed_params = params_dict)
            features = to_numpy(features)
            features = features.T
            
            # np.isnan returns an array of bools, if one value is true (there is a nan) the sum will not be zero
            assert np.isnan(features).sum() == 0, f"nan value found in audio {audio_count}, params {params_count}"
        



def get_nan_audio():
    """
    returns list of audio files to test in test_for_nan_values
    """
    audio_files = [
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-other-500/6689/64286/6689-64286-0001.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-other-500/1760/143006/1760-143006-0080.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-other-500/1579/128155/1579-128155-0012.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-other-500/25/123319/25-123319-0055.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-other-500/1579/128155/1579-128155-0012.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-other-500/4484/37119/4484-37119-0018.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-other-500/3433/135988/3433-135988-0038.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/887/123289/887-123289-0001.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/4137/11701/4137-11701-0029.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-other-500/937/148985/937-148985-0022.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-other-500/4931/28257/4931-28257-0042.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/8312/279790/8312-279790-0043.wav",
     "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-other-500/4741/27757/4741-27757-0051.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-other-500/8302/281331/8302-281331-0012.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/32/21631/32-21631-0012.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-other-500/2541/159352/2541-159352-0044.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/2092/145706/2092-145706-0022.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-other-500/5045/1197/5045-1197-0002.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-other-500/3871/692/3871-692-0025.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-other-500/1690/142293/1690-142293-0035.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-clean-100/7078/271888/7078-271888-0058.wav",
    "/mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/train-other-500/5220/112590/5220-112590-0002.wav"
    ]
    return audio_files

def get_nan_parameters():
    """
    returns a list of dictionaries for different parameters values to test
    """

    params_list = list()

    params_list.append({"W": 20,
                        "point_to_warp": 0.48068901896476746, 
                        "dist_to_warp": -4, 
                        "frequency_masking_para": 20,
                        "f": [16, 4, 1], 
                        "f0": [233, 140, 145],
                        "time_masking_para": 20, 
                        "t": [17, 7, 11],
                        "t0": [166, 841, 133]})
    params_list.append({"W": 20,
                        "point_to_warp": -0.6490881443023682,
                        "dist_to_warp": 6, 
                        "frequency_masking_para": 20,
                        "f": [16, 4, 1], 
                        "f0": [217, 94, 89],
                        "time_masking_para": 20,
                        "t": [17, 7, 11],
                        "t0": [696, 750, 637]})
    params_list.append({"W": 20,
                        "point_to_warp": 1.2762140035629272,
                        "dist_to_warp": -3,
                        "frequency_masking_para": 20,                        
                        "f": [16, 4, 1],
                        "f0": [209, 62, 97],
                        "time_masking_para": 20,
                        "t": [17, 7, 11],
                        "t0": [572, 617, 509]})
    params_list.append({"W": 20,
                        "point_to_warp": 1.745152473449707,
                        "dist_to_warp": 8,
                        "frequency_masking_para": 30,
                        "f": [24, 1],
                        "f0": [100, 87],
                        "time_masking_para": 30,
                        "t": [18, 1],
                        "t0": [186, 250]})
    params_list.append({"W": 20,
                        "point_to_warp": -0.2539668679237366,
                        "dist_to_warp": 12,
                        "frequency_masking_para": 30,
                        "f": [24, 1],
                        "f0": [96, 241],
                        "time_masking_para": 30,
                        "t": [18, 1],
                        "t0": [240, 443]})
    params_list.append({"W": 20,
                        "point_to_warp": 2.012641429901123,
                        "dist_to_warp": -14,
                        "frequency_masking_para": 30,
                        "f": [24, 1],
                        "f0": [229, 109],
                        "time_masking_para": 30,
                        "t": [18, 1],
                        "t0": [354, 891]})
    params_list.append({"W": 20,
                        "point_to_warp": 2.012641429901123,
                        "dist_to_warp": -14,
                        "frequency_masking_para": 30,
                        "f": [24, 1],
                        "f0": [96, 241],
                        "time_masking_para": 30,
                        "t": [18, 1],
                        "t0": [240, 443]})
    params_list.append({"W": 20,
                        "point_to_warp": -0.2539668679237366,
                        "dist_to_warp": 12,
                        "frequency_masking_para": 30,
                        "f": [24, 1],
                        "f0": [229, 109],
                        "time_masking_para": 30,
                        "t": [18, 1],
                        "t0": [354, 891]})

    return params_list
