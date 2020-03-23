
def onnx_coreml_paths(model_name):
    onnx_path = "./onnx_models/"+model_name+"_model.onnx"
    coreml_path = "./coreml_models/"+model_name+"_model.mlmodel"
    return onnx_path, coreml_path


def pytorch_onnx_paths(model_name):
    torch_path = "./torch_models/"+model_name+"_model.pth"
    onnx_path = "./onnx_models/"+model_name+"_model.onnx"
    config_path = "./config/"+model_name+"_config.json"

    return torch_path, config_path, onnx_path

def validation_paths(model_name):
    
    torch_path, config_path, onnx_path = pytorch_onnx_paths(model_name)
    _, coreml_path = onnx_coreml_paths(model_name)
    preproc_path = "./preproc/"+model_name+"_preproc.pyc"
    state_dict_path = "./torch_models/"+model_name+"_state_dict.pth"
    

    return torch_path, onnx_path, coreml_path, config_path, preproc_path, state_dict_path