
def onnx_coreml_paths(model_name):
    onnx_path = "./onnx_models/"+model_name+".onnx"
    coreml_path = "./coreml_models/"+model_name+".mlmodel"
    return onnx_path, coreml_path


def pytorch_onnx_paths(model_name):
    torch_path = "./torch_models/"+model_name+".pth"
    onnx_path = "./onnx_models/"+model_name+".onnx"

    return torch_path, onnx_path
