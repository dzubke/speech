# third-party libraries
import json
import pytest
# project libraries
from speech.loader import Preprocessor, AudioDataset
from speech.utils.config import Config

def test_main():
    config_json = "./cv-val_ctc-config.json"
    with open(config_json, 'r') as fid:
        config = json.load(fid)
    data_cfg = config['data']
    print(config)
    logger = None
    preproc = Preprocessor(data_cfg['dev_set'], config['preproc'], logger)
    preproc.update()
    print("preproc: \n", preproc)

    check_empty_filename(preproc)
    check_run_from_AudioDataset(preproc, data_cfg['dev_set'])

def check_empty_filename(preproc):
    wave_file = ""
    text = ['ah']
    with pytest.raises(RuntimeError) as execinfo:
        preproc.preprocess(wave_file, text)
        print(execinfo)



def check_run_from_AudioDataset(preproc:Preprocessor, data_json:str):
    """
    Runs the preprocess methood in the Preprocessor object
    over the dataset specified in config_json
    """
    audio_dataset=AudioDataset(data_json, preproc, batch_size=8)
    
    index_count = 0
    for index in range(len(audio_dataset.data)):
        audio_dataset[index]
        index_count += 1
    assert index_count == len(audio_dataset.data)

#def pytest_addoption(parser):
#    parser.addoption("--json_path", type=str, 
#         help="A json file of a dataset.")

#def pytest_generate_tests(metafunc):
#    # This is called for every test. Only get/set command line arguments
#    # if the argument is specified in the list of test "fixturenames".
#    option_value = metafunc.config.option.name
#    if 'name' in metafunc.fixturenames and option_value is not None:
#        metafunc.parametrize("name", [option_value])
