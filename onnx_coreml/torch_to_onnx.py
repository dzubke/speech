import argparse
import json

import torch

import speech.loader as loader
import speech.models as models
from get_paths import pytorch_onnx_paths
from get_test_input import generate_test_input
from import_export import torch_load, torch_onnx_export


def onnx_export(model_name, use_state_dict):  
    torch_path, onnx_path = pytorch_onnx_paths(model_name)
    torch_device = 'cpu'
    
    if use_state_dict=='True':
        state_dict_path = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/onnx_coreml/validation_scripts/state_params_20200121-0127.pth'
        config_path = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/onnx_coreml/validation_scripts/ctc_config_20200121-0127.json'
        with open(config_path, 'r') as fid:
            config = json.load(fid)
            model_cfg = config['model']
        ctc_model = models.CTC(161, 40, model_cfg) 
        state_dict = torch.load(state_dict_path)    
        ctc_model.load_state_dict(state_dict)
    
    else: 
        print(f'loaded model from: {torch_path}')
        ctc_model = torch_load(torch_path, torch_device)
        
    
    ctc_model.eval()    
    
    input_tensor = generate_test_input("pytorch", model_name, set_device=torch_device) 
    print("before torch_onnx export")
    torch_onnx_export(ctc_model, input_tensor, onnx_path)
    print(f"Torch model sucessfully converted to Onnx at {onnx_path}")

def main(model_name, use_state_dict):
    print(f'\nuse_state_dict: {use_state_dict}')

    onnx_export(model_name, use_state_dict)



if __name__ == "__main__":
    # commmand format: python pytorch_to_onnx.py <model_name>
    parser = argparse.ArgumentParser(description="converts models in pytorch to onnx.")
    parser.add_argument("model_name", help="name of the model.")
    parser.add_argument("--use_state_dict", help="boolean whether to load model from state dict") 
    args = parser.parse_args()

    main(args.model_name, args.use_state_dict)
