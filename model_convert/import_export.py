import pickle
import json

import torch
import onnx
from onnx_coreml import convert
import coremltools

import speech.models as models

def torch_load(model_path, torch_device):
    return torch.load(model_path, map_location=torch.device(torch_device))

def torch_onnx_export(torch_model, input_tensor, onnx_path, 
                    export_params=True,
                    verbose=False, 
                    opset_version=9, 
                    do_constant_folding=True,
                    input_names = ['input', 'hidden_prev', 'cell_prev'],
                    output_names = ['output', 'hidden', 'cell'],
                    dynamic_axes = None
                    ):

    torch.onnx.export(torch_model,               # model being run
                input_tensor,              # model input (or a tuple for multiple inputs)
                onnx_path,   # where to save the model (can be a file or file-like object)
                #export_params=export_params,  # store the trained parameter weights inside the model file
                opset_version=opset_version,          # the ONNX version to export the model to
                do_constant_folding=do_constant_folding,  # whether to execute constant folding for opti
                input_names = input_names,   # the model's input names
                output_names = output_names, # the model's output names
                #dynamic_axes=dynamic_axes)    # variable lenght axes
                )


def onnx_coreml_export(onnx_path, coreml_path):
    onnx_model = onnx.load(onnx_path)
    
    coreml_model = convert(
        model=onnx_model,
        minimum_ios_deployment_target = '13'
    )
    coreml_model.save(coreml_path)


def preproc_to_dict(preproc_path_in, preproc_path_out=None, export=False):
    with open(preproc_path_in, 'rb') as fid:
        preproc = pickle.load(fid)
        preproc_dict = {'mean':preproc.mean.tolist(),
                        'std': preproc.std.tolist(),
                        "_input_dim": preproc._input_dim,
                        "start_and_end": preproc.start_and_end,
                        "int_to_char": preproc.int_to_char,
                        "char_to_int": preproc.char_to_int
                        }

        if export:
            with open(preproc_path_out, 'wb') as fid:
                pickle.dump(preproc_dict, fid)
        else:
            return preproc_dict


def preproc_to_json(preproc_path, json_path):
    preproc_dict = preproc_to_dict(preproc_path, export=False)

    with open(json_path, 'w') as fid:
        json.dump(preproc_dict, fid)

def export_metadata_json(json_path:str):
    pass


def export_state_dict(model_path, state_dict_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    torch.save(model.state_dict(), state_dict_path)


def export_torch_model():
    state_dict_path = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/onnx_coreml/validation_scripts/state_params_20200121-0127.pth' 
    cfg_path ='/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/onnx_coreml/validation_scripts/ctc_config_20200121-0127.json'
    new_model_path = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/onnx_coreml/torch_models/20200121-0127_best_model_pyt14.pth'
    with open(cfg_path, 'r') as fid:
        config = json.load(fid)
        model_cfg = config['model']
        
    ctc_model = models.CTC(161, 40, model_cfg)
    state_dict = torch.load(state_dict_path)
    ctc_model.load_state_dict(state_dict)
    torch.save(ctc_model, new_model_path)
