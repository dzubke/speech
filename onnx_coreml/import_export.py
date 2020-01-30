import sys

from torch import load, onnx, device
from onnx import onnx_pb
from onnx_coreml import convert
import coremltools

from model_generator import TestNet
import torch.nn as nn

def torch_load(model_path, torch_device):
    return load(model_path, map_location=device(torch_device))

def torch_onnx_export(torch_model, input_tensor, onnx_path, 
                    export_params=True,
                    verbose=True, 
                    opset_version=9, 
                    do_constant_folding=False,
                    input_names = ['input'],
                    output_names = ['output'],
                    keep_initializers_as_inputs=True, 
                    dynamic_axes = {'input' : {0 : 'batch_size', 1: 'time_dim'}, 'output' : {0 : 'batch_size', 1: 'time_dim'}}
                    ):
    """
    """

    onnx.export(torch_model,               # model being run
                      input_tensor,              # model input (or a tuple for multiple inputs)
                      onnx_path,   # where to save the model (can be a file or file-like object)
                      export_params=export_params,  # store the trained parameter weights inside the model file
                      opset_version=opset_version,          # the ONNX version to export the model to
                      do_constant_folding=do_constant_folding,  # whether to execute constant folding for optimization
                      input_names = input_names,   # the model's input names
                      output_names = output_names, # the model's output names
                      dynamic_axes=dynamic_axes)    # variable lenght axes


def onnx_coreml_export(onnx_path, coreml_path):
    print("mark 1")
    model_file = open(onnx_path, 'rb')
    print("mark 2")
    model_proto = onnx_pb.ModelProto()
    print("mark 3")
    model_proto.ParseFromString(model_file.read())
    print("mark 4")
    coreml_model = convert(
        model_proto,
        minimum_ios_deployment_target = '13'
    )
    print("mark 5")
    coreml_model.save(coreml_path)

def onnx_coreml_export_2(model_in, model_out):
    mlmodel = convert(model=model_in, minimum_ios_deployment_target='13')

    # rename inputs/outputs
    spec = mlmodel.get_spec()
    coremltools.utils.rename_feature(spec, current_name='input.1', new_name='input_tensor')
    coremltools.utils.rename_feature(spec, current_name='12', new_name='network_output')
    mlmodel = coremltools.models.MLModel(spec)

    # Save converted CoreML model
    mlmodel.save(model_out)

