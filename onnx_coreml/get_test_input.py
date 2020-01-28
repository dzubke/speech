import torch
import numpy


def generate_test_input(model_format:str ,model_name:str, set_device=None ):
    """outputs a test input based on the model format ("pytorch" or "onnx") and the model name
    """
    batch_size = 1
    
    device = ".cuda()" if torch.cuda.is_available() and set_device!='cpu'  else ""
    
    if model_format == "pytorch":       
        if model_name == "super_resolution":
            return eval("torch.randn(batch_size, 1, 224, 224, requires_grad=True)"+device)
        elif model_name == "resnet18" or model_name == "alexnet":
            return eval("torch.randn(batch_size, 3, 224,224, requires_grad=True)"+device)
        else:
            return eval("torch.FloatTensor(batch_size, 200, 161)"+device)
                #eval("torch.IntTensor(batch_size, 41)"+device)]

    elif model_format == "onnx":
        if model_name == "super_resolution":
            raise NotImplementedError
        elif model_name == "resnet18" or  "alexnet":
            return numpy.random.randn(batch_size, 3, 224,224)
        else:
            raise NotImplementedError

    else: 
        raise ValueError("model_format parameters must be 'pytorch' or 'onnx'")
