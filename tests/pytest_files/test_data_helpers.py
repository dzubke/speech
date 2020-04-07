import pytest

from speech.utils.data_helpers import skip_file


def test_skip_file():
    true_cases = [
        # lower bound of error files
        ("tatoeba",  "/home/dzubke/awni_speech/data/tatoeba/tatoeba_audio_eng/audio/CK/6122903.wv"),
        # upper bound of error files
        ("tatoeba", "/home/dzubke/awni_speech/data/tatoeba/tatoeba_audio_eng/audio/CK/6123834.wv"),
        # inside bounds
        ("tatoeba", "/home/dzubke/awni_speech/data/tatoeba/tatoeba_audio_eng/audio/CK/6123777.wv"),
        ("voxforge", "/home/dzubke/awni_speech/data/voxforge/archive/DermotColeman-20111125-uom/wav/b0396.wv")
    ]
    false_cases = [
        # outside lower bound
        ("tatoeba", "/home/dzubke/awni_speech/data/tatoeba/tatoeba_audio_eng/audio/CK/6122902.wv"),
        # outside upper bound
        ("tatoeba", "/home/dzubke/awni_speech/data/tatoeba/tatoeba_audio_eng/audio/CK/6123835.wv"),
        ("voxforge", "/home/dzubke/awni_speech/data/voxforge/archive/DermotColeman-20111125-uom/wav/b0395.wv"),
        ("librispeech", ""),
        ("tedlium","")
    ]
    valueerror_cases = [
        ("not_dataset_name", "")
    ]
    for case in true_cases:
        assert skip_file(*case) == True, "case should be True"
    for case in false_cases:
        assert skip_file(*case) == False, "case should be False"
    with pytest.raises(ValueError):
        for case in valueerror_cases:
            skip_file(*case)
