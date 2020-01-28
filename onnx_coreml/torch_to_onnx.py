import torch
import argparse

from get_paths import pytorch_onnx_paths
from get_test_input import generate_test_input
from import_export import torch_load, torch_onnx_export


def onnx_export(model_name):

    torch_path, onnx_path = pytorch_onnx_paths(model_name)
    
    torch_device = 'cpu'

    torch_model = torch_load(torch_path, torch_device)
    torch_model.eval()    
    #dummy_tensor = (torch.from_numpy(dummy_input[0][0]).float().to('cpu'), [27])
    #dummy_tensor = [torch.FloatTensor(1, 200, 161), torch.IntTensor(1, 41)]
    
    input_tensor = generate_test_input("pytorch", model_name, set_device=torch_device) 
    torch_onnx_export(
        torch_model, input_tensor, onnx_path, verbose=True
        )

def main(model_name):
    onnx_export(model_name)



if __name__ == "__main__":
    # commmand format: python pytorch_to_onnx.py <model_name>
    parser = argparse.ArgumentParser(description="converts models in pytorch to onnx.")
    parser.add_argument("model_name", help="name of the model.")
    args = parser.parse_args()

    main(args.model_name)
