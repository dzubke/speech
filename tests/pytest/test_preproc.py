# third-party libraries
import pytest
# project libraries
from speech.loader import Preprocessor, AudioDataset
from speech.utils.config import Config

def test_main():
    config_json = ""
    run_preprocess(config_json)


def run_preprocess(config_jsjon:str):
    """
    Runs the preprocess methood in the Preprocessor object
    over the dataset specified in config_json
    """
    config = Config(config_json)
    data_json = config.data_cfg.get("train_set")
    logger = None
    preproc = Preprocessor(data_json, config.preproc_cfg, logger)
    audio_dataset=AudioDataset(data_json, preproc, batch_size=8)

    for index in range(len(audio_dataset.data)):
        print(audio_dataset[index])
        assert audio_dataset[index] == True

def pytest_addoption(parser):
    parser.addoption("--json_path", type=str, 
         help="A json file of a dataset.")

def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.name
    if 'name' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("name", [option_value])