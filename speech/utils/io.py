import json 
import os
import pickle
import torch

MODEL = "model"
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

