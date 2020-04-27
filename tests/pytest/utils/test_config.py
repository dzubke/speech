
# third-party libaries
import pytest
# project libraries
from speech.utils.config import Config



def test_config():
    json_path = "./test_ctc_config.json"
    config = Config(json_path)

    assert config.config.get("seed") == 2017
    assert config.config.get("save_path") == "/home/dzubke/awni_speech/speech/examples/librispeech/models/ctc_models/20200422/"
    assert config.data_cfg.get("train_set") =="/home/dzubke/awni_speech/data/lib-ted-cv/train_lib-ted-cv.json"
    assert config.data_cfg.get("start_and_end") == False
    assert config.log_cfg.get("use_log") == True
    assert config.preproc_cfg.get("preprocessor") == "log_spec"
    assert config.preproc_cfg.get("window_size") == 32
    assert config.preproc_cfg.get("use_spec_augment") == True
    assert config.preproc_cfg.get("noise_levels") == [0,0.7]
    assert config.opt_cfg.get("dampening") == 0.98
    assert config.model_cfg.get("encoder").get("rnn").get("type") == "LSTM"
    assert config.model_cfg.get("encoder").get("rnn").get("dim") == 512
    assert config.model_cfg.get("encoder").get("rnn").get("bidirectional") == False




