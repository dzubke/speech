import argparse
import json
import os

import torch

import speech.loader as loader
import speech.models as models
from get_paths import pytorch_onnx_paths
from get_test_input import generate_test_input
from import_export import torch_load, torch_onnx_export


def onnx_export(model_name, num_frames, use_state_dict):  
    
    freq_dim = 257  #freq dimension out of log_spectrogram 
    time_dim = num_frames

    torch_path, config_path, onnx_path = pytorch_onnx_paths(model_name)

    torch_device = 'cpu'
    
    if use_state_dict=='True':
        print(f'loaded state_dict from: {torch_path}')
        
        with open(config_path, 'r') as fid:
            config = json.load(fid)
            model_cfg = config['model']
        
        ctc_model = models.CTC(freq_dim, 39, model_cfg) 
        state_dict_model = torch.load(torch_path, map_location=torch.device(torch_device))
        ctc_model.load_state_dict(state_dict_model.state_dict())
    
    else: 
        print(f'loaded entire model from: {torch_path}')
        ctc_model = torch.load(torch_path, map_location=torch.device(torch_device))
        torch.save(ctc_model, torch_path)
        ctc_model = torch.load(torch_path, map_location=torch.device(torch_device))

        
    
    ctc_model.eval()    
    
    input_tensor = generate_test_input("pytorch", model_name, time_dim=time_dim, set_device=torch_device) 
    torch_onnx_export(ctc_model, input_tensor, onnx_path)
    print(f"Torch model sucessfully converted to Onnx at {onnx_path}")

def main(model_name, num_frames, use_state_dict):
    print(f'\nuse_state_dict: {use_state_dict}')

    onnx_export(model_name, num_frames, use_state_dict)



if __name__ == "__main__":
    # commmand format: python pytorch_to_onnx.py <model_name> --num_frames X --use_state_dict <True/False>
    parser = argparse.ArgumentParser(description="converts models in pytorch to onnx.")
    parser.add_argument("model_name", help="name of the model.")
    parser.add_argument("--num_frames", help="number of input frames in time dimension hard-coded in onnx model")
    parser.add_argument("--use_state_dict", help="boolean whether to load model from state dict") 
    args = parser.parse_args()

    main(args.model_name, int(args.num_frames), args.use_state_dict)
