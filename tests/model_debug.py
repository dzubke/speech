# standard libraries
import json

#third party libraries
import torch
import torch.nn as nn
import numpy as np

# project libraries
from speech import loader
from speech.utils import wave
from speech.utils.io import load
from train import check_nan
import functions.ctc as ctc


MODEL_PATH = "/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/examples/librispeech/models/ctc_models/20200305/20200308"
CONFIG_PATH = "/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/examples/librispeech/models/ctc_models/20200305/20200308/ctc_config.json"
AUDIO_PATH = "/Users/dustin/CS/consulting/firstlayerai/data/LibriSpeech/dev-clean/174/84280/174-84280-0000.wav"
TEXT = ["hh", "aw", "w", "iy", "m", "ah", "s", "t", "s", "ih", "m", "p", "l", "ah", "f", "ay"]

def main():
    tests = ["no_alter", "zero",  "all_nan", "one_nan"]

    for test in tests:
        print(f"===================================")
        print(f"test: {test}")
        test_model(test)
        print(f"===================================\n")


def test_model(test_label):
    """
        this checks if there are nans in a variety of test scenarios
        Arguments:
        test_label (str): the label passed into alter_logspec
    """

    with open(CONFIG_PATH, 'r') as fid:
        config = json.load(fid)

    opt_cfg = config["optimizer"]
    model_cfg = config["model"]

    model, preproc = load(MODEL_PATH)

    optimizer = torch.optim.SGD(model.parameters(),
                    lr=opt_cfg["learning_rate"],
                    momentum=opt_cfg["momentum"])


    log_spec, samp_rate = load_audio(AUDIO_PATH, preproc)

    inputs = alter_logspec(log_spec, alter=test_label)
    labels = [preproc.encode(TEXT)]

    batch = ((inputs, labels))
    x, y, x_lens, y_lens = model.collate(*batch)
    out, rnn_args = model.forward_impl(x)
    loss_fn = ctc.CTCLoss()
    loss = loss_fn(out, y, x_lens, y_lens)
    loss.backward()
    loss = loss.item()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 200)
    print(f"loss: {loss}, grad_norm: {grad_norm}")
    optimizer.step()

    print(f"Are there nan's? : {check_nan(model)}")


def load_audio(audio_path, preproc):
    
    audio_data, samp_rate = wave.array_from_wave(audio_path)
    inputs = loader.log_specgram_from_data(audio_data, samp_rate, 
                window_size=32, step_size=16)
    inputs = (inputs - preproc.mean) / preproc.std
    return inputs, samp_rate

def alter_logspec(log_spec, alter):
    """
        processes the log_spec in a certain way to test if the model fails
    """

    if alter == 'zero':
        log_spec[:, :]= 0 
    elif alter == 'all_nan':
        log_spec[:,:] = np.nan
    elif alter == 'one_nan':
        log_spec[10,10] = np.nan
        
    return np.expand_dims(log_spec, axis=0)


if __name__ == "__main__":
     main()
