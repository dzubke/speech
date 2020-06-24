# third party libraries
import pytest
from _pytest.fixtures import SubRequest
# project libraries
from speech.utils.wave import array_from_wave
from speech.loader import log_spectrogram_from_data
from tests.pytest.utils import get_all_test_audio

def test_single_values():
    audio_paths = get_all_test_audio()
    for audio_path in audio_paths:
        audio_data, samp_rate = array_from_wave(audio_path)
        log_spec = log_spectrogram_from_data(audio_data, samp_rate, window_size=32, step_size=16)
    raise NotImplementedError
        # normalize
        # spec_aug



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

