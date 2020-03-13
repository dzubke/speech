import torch
import numpy


def generate_test_input(model_format:str ,model_name:str, time_dim: int, set_device=None ):
    """outputs a test input based on the model format ("pytorch" or "onnx") and the model name
        
        Arguments
            time_dim: time_dimension into the model
    """
    batch_size = 1
    layer_count = 5 
    
    device = ".cuda()" if torch.cuda.is_available() and set_device!='cpu'  else ""
    
    if model_format == "pytorch":       
        if model_name == "super_resolution":
            return eval("torch.randn(batch_size, 1, 224, 224, requires_grad=True)"+device)
        elif model_name == "resnet18" or model_name == "alexnet":
            return eval("torch.randn(batch_size, 3, 224,224, requires_grad=True)"+device)
        elif model_name == "lstm":
            return (eval("torch.randn(5, 3, 10)"+device), 
                    eval("torch.randn(layer_count * 2, 3, 20)"+device),
                    eval("torch.randn(layer_count * 2, 3, 20)"+device) 
                    )
        else:
            return (eval("torch.randn(1,time_dim, 257)"+device),
                    (eval("torch.randn(layer_count * 1, 1, 512)"+device),
                    eval("torch.randn(layer_count * 1, 1, 512)"+device))
                    )

    elif model_format == "onnx":
        if model_name == "super_resolution":
            raise NotImplementedError
        elif model_name == "resnet18" or  "alexnet":
            return numpy.random.randn(batch_size, 3, 224,224)
        else:
            raise NotImplementedError

    else: 
        raise ValueError("model_format parameters must be 'pytorch' or 'onnx'")
