#some standard imports
import io
import argparse
import numpy as np

import torch

import onnx
import onnxruntime

from get_paths import pytorch_onnx_paths
from get_test_input import generate_test_input
from import_export import torch_load, torch_onnx_export

from SimpleNet import SimpleNet


def test_onnxruntime(model_name):
    """this function takes in a pytorch_model as input
    """    
    
    #input_tensor = torch.randn(5, 3, 10, requires_grad=True).cuda()
    #input_tensor = generate_test_input("pytorch", model_name)
    input_tensor, h0, c0 = generate_test_input("pytorch", model_name)    

    torch_path, onnx_path = pytorch_onnx_paths(model_name)
    
    torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    torch_model = torch_load(torch_path, torch_device)
    torch_model.eval()
    #torch_output = torch_model(input_tensor)
    torch_output, (hn, cn) = torch_model(input_tensor, (h0, c0))    
    print(f"torch_output: {torch_output}")   
 
    #torch_output = torch_export_inference(torch_path, onnx_path, input_tensor)
    
    onnx_runtime(input_tensor, onnx_path, torch_output)

def torch_export_inference(torch_path: str, onnx_path:str, input_tensor):
    """ DEPRICATED --> THIS FUNCTION ISN'T USED. I JUST HAVEN'T DELETED IT YOU. ITS HARD TO LET GO SOMETIMES...        
        takes in a path to a pytorch model, loads the model, conducts inference on the input_tensor
        and exports the pytorch model as an onnx model. the method outputs the torch_ouput tensor that contains
        the inference
    """

    torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    torch_model = torch.load(torch_path, map_location=torch.device(torch_device))
    print(f"torch_model: {type(torch_model)}")


    # set the model to inference mode
    torch_model.eval()


    #------------------------------------------
    # Input to the model
    torch_output = torch_model(input_tensor)
    print(f"torch_output: {torch_output}")
    
    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      input_tensor,              # model input (or a tuple for multiple inputs)
                      onnx_path,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=9,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size', 1: 'time_dim'},    # variable lenght axes
                                    'output' : {0 : 'batch_size', 1: 'time_dim'}})
    
    return torch_output


def to_numpy(tensor):
    if isinstance(tensor, list):
        tensor_list = []
        for _tensor in tensor:
            numpy_tensor = _tensor.detach().cpu().numpy() if _tensor.requires_grad else _tensor.cpu().numpy()
            tensor_list.append(numpy_tensor)
        return tensor_list

    else:
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def onnx_runtime(input_tensor, onnx_path, torch_output):

    onnx_model = onnx.load(onnx_path)
    print(f"onnx model type: {type(onnx_model)}")
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_path)
    
    # compute ONNX Runtime output prediction
    print(f"ort_session.get_inputs(): {ort_session.get_inputs()}")
    if isinstance(input_tensor, list):
        ort_inputs = {"inputs": to_numpy(input_tensor[0]), "labels": to_numpy(input_tensor[1])}
    else:
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(f"{ort_outs}, {np.shape(ort_outs)}")

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_output), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def main(model_name):
    #convert_pytorch_onnx()
    test_onnxruntime(model_name)



if __name__ == "__main__":
    # commmand format: python onnx_runtime.py <model_name>
    parser = argparse.ArgumentParser(description="paths to test onnxruntime.")
    parser.add_argument("model_name", help="name of the model.")
    args = parser.parse_args()

    main(args.model_name)    
