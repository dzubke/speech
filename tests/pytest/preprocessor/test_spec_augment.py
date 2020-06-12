# third party libraries
import pytest
from _pytest.fixtures import SubRequest
# project libraries
from speech.utils.wave import array_from_wave
from speech.loader import log_specgram_from_data, sample_normalize


def test_single_values():
    audio_path = "/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/tests/pytest/test_audio/Speak-out.wav"
    audio_data, samp_rate = array_from_wave(wave_file)
    log_spec = log_specgram_from_data(audio_data, samp_rate, window_size=32, step_size=16)
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

