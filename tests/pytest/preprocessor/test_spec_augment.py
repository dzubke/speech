# standard libraries
from logging import Logger

# third party libraries
import pytest
from _pytest.fixtures import SubRequest
# project libraries
from speech.loader import log_spectrogram_from_data
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


def test_logger():
    """
    Runs all the tests with a logger to test if logger fails
    """
    pass


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

