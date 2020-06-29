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
                
                print(f"number of masked rows: {num_mask_rows}, max_masked: {max_freq_masks}")
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
                
                print(f"number of time masked rows: {num_mask_rows}, max_time_masked: {max_time_masks}")
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


# @pytest.fixture
# def wallet(request: SubRequest):
#    param = getattr(request, ‘param’, None)
#    if param:
#      prepared_wallet = Wallet(initial_amount=param[0])
#    else:
#      prepared_wallet = Wallet()
#    yield prepared_wallet
#    prepared_wallet.close()


# def test_default_initial_amount(wallet):
#    assert wallet.balance == 0
   
# @pytest.mark.parametrize(‘wallet’, [(100,)], indirect=True)
# def test_setting_initial_amount(wallet):
#    assert wallet.balance == 100

