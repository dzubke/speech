import torch
import argparse

from get_paths import pytorch_onnx_paths
from get_test_input import generate_test_input

def onnx_export(model_name):

    pytorch_path, onnx_path = pytorch_onnx_paths(model_name)
    
    torch_device = 'cpu'

    torch_model = torch.load(pytorch_path, map_location=torch.device(torch_device))
    torch_model.eval()    
    #dummy_tensor = (torch.from_numpy(dummy_input[0][0]).float().to('cpu'), [27])
    #dummy_tensor = [torch.FloatTensor(1, 200, 161), torch.IntTensor(1, 41)]
    
    input_tensor = generate_test_input("pytorch", model_name, set_device=torch_device) 
    onnx_model = torch.onnx.export(
        torch_model, input_tensor, onnx_path, input_names=['data'],
        output_names = ['output'], verbose=True
        )

def main(model_name):
    onnx_export(model_name)



if __name__ == "__main__":
    # commmand format: python pytorch_to_onnx.py <model_name>
    parser = argparse.ArgumentParser(description="converts models in pytorch to onnx.")
    parser.add_argument("model_name", help="name of the model.")
    args = parser.parse_args()

    main(args.model_name)
