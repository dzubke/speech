# standard libraries
from collections import OrderedDict
import json 
import os
import pickle
import yaml
# third-party libraries
import torch

MODEL = "model.pth"
PREPROC = "preproc.pyc"

def get_names(path, tag):
    tag = tag + "_" if tag else ""
    model = os.path.join(path, tag + MODEL)
    preproc = os.path.join(path, tag + PREPROC)
    return model, preproc

def save(model, preproc, path, tag=""):
    model_n, preproc_n = get_names(path, tag)
    torch.save(model, model_n)
    with open(preproc_n, 'wb') as fid:
        pickle.dump(preproc, fid)

def load(path, tag=""):
    model_n, preproc_n = get_names(path, tag)
    model = torch.load(model_n, map_location=torch.device('cpu'))
    with open(preproc_n, 'rb') as fid:
        preproc = pickle.load(fid)
    return model, preproc

def load_pretrained(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    return model

def save_dict(dct, path):
    with open(path, 'wb') as fid:
        pickle.dump(dct, fid)

def export_state_dict(model_in_path, params_out_path):
    model = torch.load(model_in_path, map_location=torch.device('cpu'))
    pythtorch.save(model.state_dict(), params_out_path)

def read_data_json(data_path):
    with open(data_path) as fid:
        return [json.loads(l) for l in fid]

def write_data_json(dataset:list, write_path:str):
    """
    Writes a list of dictionaries in json format to the write_path
    """
    with open(write_path, 'w') as fid:
        for example in dataset:
            json.dump(example, fid)
            fid.write("\n")

def read_pickle(pickle_path:str):
    assert pickle_path != '', 'pickle_path is empty'
    with open(pickle_path, 'rb') as fid:
        pickle_object = pickle.load(fid)
    return pickle_object

def write_pickle(pickle_path:str, object_to_pickle):
    assert pickle_path != '', 'pickle_path is empty'
    with open(pickle_path, 'wb') as fid:
        pickle.dump(object_to_pickle, fid) 

def load_config(config_path:str)->dict:
    """
    loads the config file in json or yaml format
    """
    _, config_ext = os.path.splitext(config_path)    

    if config_ext == '.json':
        with open(config_path, 'r') as fid:
            config = json.load(fid)
    elif config_ext == '.yaml':
        with open(config_path, 'r') as config_file:
            config = yaml.load(config_file) 
    else:
        raise ValueError(f"config file extension {config_ext} not accepted")
    
    return config

def load_from_trained(model, model_cfg):
    """
    loads the model with pretrained weights from the model in
    model_cfg["trained_path"]
    Arguments:
        model (torch model)
        model_cfg (dict)
    """
    trained_model = torch.load(model_cfg["trained_path"], map_location=torch.device('cpu'))
    trained_state_dict = trained_model.state_dict()
    trained_state_dict = filter_state_dict(trained_state_dict, remove_layers=model_cfg["remove_layers"])
    model_state_dict = model.state_dict()
    model_state_dict.update(trained_state_dict)
    model.load_state_dict(model_state_dict)
    return model


def filter_state_dict(state_dict, remove_layers=[]):
    """
    filters the inputted state_dict by removing the layers specified
    in remove_layers
    Arguments:
        state_dict (OrderedDict): state_dict of pytorch model
        remove_layers (list(str)): list of layers to remove 
    """

    state_dict = OrderedDict(
        {key:value for key,value in state_dict.items()
        if key not in remove_layers}
        )
    return state_dict

