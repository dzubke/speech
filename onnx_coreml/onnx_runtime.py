#some standard imports
import io
import argparse
import numpy as np

import torch
import torch.onnx

import onnx
import onnxruntime

from get_paths import pytorch_onnx_paths
from get_test_input import  generate_test_input
from import_export import torch_load, torch_onnx_export



def test_onnxruntime(model_name):
    """this function takes in a pytorch_model as input
    """
    
    input_tensor = generate_test_input("pytorch", model_name)
    
    torch_path, onnx_path = pytorch_onnx_paths(model_name)

    
    torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch_model = torch_load(torch_path, torch_device) 
    torch_model.eval()
    
    torch_output = torch_model(input_tensor)

    torch_onnx_export(torch_model, input_tensor, onnx_path, export_params=True, opset_version=9, 
                    do_constant_folding=True, input_names = ['input'],
                    output_names = ['output'], dynamic_axes = None
    )
    
    ort_outs = onnx_runtime(input_tensor, onnx_path, torch_output)
    
    print(f"torch output: {torch_output}")
    print(f"onnx output: {ort_outs}")

def to_numpy(tensor):
    if isinstance(tensor, list):
        tensor_list = []
        for _tensor in tensor:
            numpy_tensor = _tensor.detach().cpu().numpy() if _tensor.requires_grad else _tensor.cpu().numpy()
            tensor_list.append(numpy_tensor)
        return tensor_list

    else:
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def onnx_runtime(infer_tensor, onnx_path, torch_output):

    onnx_model = onnx.load(onnx_path)
    print(f"onnx model type: {type(onnx_model)}")
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_path)
    
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(infer_tensor)}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_output), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    return ort_outs[0]

def main(model_name):
    #convert_pytorch_onnx()
    test_onnxruntime(model_name)



if __name__ == "__main__":
    # commmand format: python onnx_runtime.py <model_name>
    parser = argparse.ArgumentParser(description="paths to test onnxruntime.")
    parser.add_argument("model_name", help="name of the model.")
    args = parser.parse_args()

    main(args.model_name)    
