# standard libraries
import os

ROOT = "/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/model_convert"

def onnx_coreml_paths(model_name):
    onnx_path = os.path.join(ROOT, "onnx_models", model_name+"_model.onnx")
    coreml_path = os.path.join(ROOT, "coreml_models", model_name+"_model.mlmodel")

    assert os.path.exists(onnx_path), "onnx_coreml_paths: onnx path doesn't exist"
    assert os.path.exists(coreml_path), "onnx_coreml_paths: coreml path doesn't exist"

    return onnx_path, coreml_path


def pytorch_onnx_paths(model_name):
    torch_path = os.path.join(ROOT, "torch_models", model_name+"_model.pth")
    onnx_path = os.path.join(ROOT, "onnx_models", model_name+"_model.onnx")
    config_path = os.path.join(ROOT, "config", model_name+"_config.json")
    
    assert os.path.exists(torch_path), "pytorch_onnx_paths: torch path doesn't exist"
    #assert os.path.exists(onnx_path), "pytorch_onnx_paths: onnx_path doesn't exist"
    assert os.path.exists(config_path), "pytorch_onnx_paths: config path doesn't exist"

    return torch_path, config_path, onnx_path

def validation_paths(model_name):
    
    torch_path, config_path, onnx_path = pytorch_onnx_paths(model_name)
    _, coreml_path = onnx_coreml_paths(model_name)
    preproc_path = os.path.join(ROOT, "preproc", model_name+"_preproc.pyc")
    state_dict_path = os.path.join(ROOT, "torch_models", model_name+"_state_dict.pth")
    
    assert os.path.exists(preproc_path), "validation_paths: preproc path doesn't exist"
    # state_dict_path will be written to. don't need to check existence

    return torch_path, onnx_path, coreml_path, config_path, preproc_path, state_dict_path
