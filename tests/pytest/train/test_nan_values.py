# standard libraries
import json
import os 
# third-party libraries
import numpy as np
import pytest
import torch 
import torch.nn as nn
# project libraries         
from speech.models.ctc_model_train import CTC_train 
import speech.loader as loader 
from speech.utils.compat import get_main_dir_path
from speech.utils.data_structs import Batch
from speech.utils.io import read_pickle, load_from_trained
from speech.utils.model_debug import check_nan_params_grads

# setting detect_anomally to be globally true
torch.autograd.set_detect_anomaly(True)

def test_zero_stddev_saved_batch():
    """
    Loads a saved_batch that has all zeros as the last feature in the batch and
    ensure so error is raised
    """
    print("\n\033[94m starting test_zero_stdddev_saved_batch... \033[00m")
    zero_stddev_batch = get_zero_stddev_saved_batch()
    print(f"input,label size 2=={len(zero_stddev_batch)}\n \
            batch_size {len(zero_stddev_batch[0])}\n \
            first example shape:  {zero_stddev_batch[0][0].shape}")
    # check that the last feature is all zeros
    assert zero_stddev_batch[0][7].sum() == 0
    config_path = "examples/librispeech/models/ctc_models/20200625/ph5/test-ph1/ctc_config_ph1.json"

    forward_backward_prop(config_path, zero_stddev_batch)


def test_all_zero_batch():
    """
    Sets all features in a batch to zero and checks not exception is raised
    """
    print("\n\033[94m starting test_all_zero_batch...\033[00m")
    all_zero_batch = get_zero_stddev_saved_batch()
    
    # batch of 8 sets of zero arrays
    all_zero_batch[0] = tuple([np.zeros((100, 257), dtype=np.float32) for i in range (8)])
    
    config_path = "examples/librispeech/models/ctc_models/20200625/ph5/test-ph1/ctc_config_ph1.json"

    forward_backward_prop(config_path, all_zero_batch)


def test_single_nan_value_batch(check_nan=True):
    """
    sets a single value in one feature to a nan value and checks that a Runtime Error is raised
    """
    print("\n\033[94m starting test_single_nan_value_batch... \033[00m")
    single_nan_batch = get_zero_stddev_saved_batch()

    # setting the first element of the first feature in the batch to np.nan
    single_nan_batch[0][0][0,0] = np.nan

    config_path = "examples/librispeech/models/ctc_models/20200625/ph5/test-ph1/ctc_config_ph1.json"
    with pytest.raises(RuntimeError) as execinfo:
        forward_backward_prop(config_path, single_nan_batch, check_nan=check_nan)
        print(f"expected exception raised: {execinfo}")

def test_nan_in_parameters():
    """
    I want to check when/where the nan values appear in the paramaters and gradients
    """
    print("\n\033[94m starting test_nan_in_parameters...\033[00m")
    test_batch = get_zero_stddev_saved_batch()

    # setting the first element of the first feature in the batch to np.nan
    test_batch[0][0][0,0] = np.nan

    config_path = "examples/librispeech/models/ctc_models/20200625/ph5/test-ph1/ctc_config_ph1.json"

    with pytest.raises(RuntimeError):
        model, optimizer = setup_model_optim(config_path)
        print("model and optimizer created, no nans")
        optimizer.zero_grad()
        print("grads zeroed, no nans")
        loss = model.loss(test_batch)
        print("loss calculated, no nans")
        loss.backward()
        print(f"loss: {loss.item()}, no nans")

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 200)
        assert not check_nan_params_grads(model.parameters()), "nans in params or grads"
        print(f"grad_norm: {grad_norm}, no nans")

        optimizer.step()

        assert not check_nan_params_grads(model.parameters()), "nans in params or grads"


def test_no_error_with_nan():
    """
    Checks that no error is raised if anomal_detect is false
    check_nan in forward_backward_prop is set to False to remove the check_nan assert
    """
    print("\n\033[94m starting test_no_error_with_nan...\033[00m")
    with torch.autograd.set_detect_anomaly(False):
        single_nan_batch = get_zero_stddev_saved_batch()

        # setting the first element of the first feature in the batch to np.nan
        single_nan_batch[0][0][0,0] = np.nan

        config_path = "examples/librispeech/models/ctc_models/20200625/ph5/test-ph1/ctc_config_ph1.json"

        forward_backward_prop(config_path, single_nan_batch, check_nan=False)


def get_zero_stddev_saved_batch():
    main_dir = get_main_dir_path()
    saved_batch_path = os.path.join(main_dir, "saved_batch", 
                        "2020-06-25-07-06_ph1_smaller-context-NaN-debug_lib-ted-cv_batch.pickle")
    return read_pickle(saved_batch_path)


def forward_backward_prop(config_path:str, test_batch:Batch, check_nan:bool=True):
    """
    runs a forward and backward propagration pass through the model defined by the config in 
    config_path using the test_batch
    Arguments:
        check_nan - bool: determines whether nan values params and grads will be checked
    """
    model, optimizer = setup_model_optim(config_path)
    print("model and optimizer created")    
    optimizer.zero_grad()
    print("grads zeroed")
    loss = model.loss(test_batch)
    print("loss calculated")
    loss.backward()
    print(f"loss: {loss.item()}")

    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 200)
    print(f"grad_norm: {grad_norm}")

    optimizer.step()
    
    if check_nan:
        assert not check_nan_params_grads(model.parameters())


def setup_model_optim(config_path:str):
    

    with open(config_path, 'r') as fid: 
        config = json.load(fid) 

    data_cfg = config["data"] 
    preproc_cfg = config["preproc"] 
    opt_cfg = config["optimizer"] 
    model_cfg = config["model"] 

    logger = None 
    preproc = loader.Preprocessor(data_cfg["train_set"], preproc_cfg, logger,  
                       max_samples=100, start_and_end=data_cfg["start_and_end"]) 
    model = CTC_train(preproc.input_dim, 
                             preproc.vocab_size, 
                             model_cfg)
    
    if model_cfg["load_trained"]:
        model = load_from_trained(model, model_cfg)
        print(f"Succesfully loaded weights from trained model: {model_cfg['trained_path']}")
    model.cuda() if torch.cuda.is_available() else model.cpu() 

    optimizer = torch.optim.SGD(model.parameters(), 
                         lr= opt_cfg['learning_rate'], 
                         momentum=opt_cfg["momentum"], 
                         dampening=opt_cfg["dampening"]) 

    return model, optimizer
