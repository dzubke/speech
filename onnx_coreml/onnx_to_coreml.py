import sys
import argparse

from onnx import onnx_pb
from onnx_coreml import convert
import coremltools
from get_paths import onnx_coreml_paths

def converter_1(onnx_path, coreml_path, input_names, output_names):
    model_file = open(onnx_path, 'rb')
    model_proto = onnx_pb.ModelProto()
    model_proto.ParseFromString(model_file.read())
    coreml_model = convert(
        model_proto#,  input_names=input_names, output_names=output_names, 
       # minimum_ios_deployment_target = '13' 
    )
    coreml_model.save(coreml_path)

def converter_2(model_in, model_out):
    mlmodel = convert(model=model_in, minimum_ios_deployment_target='13')
      
    # rename inputs/outputs
    spec = mlmodel.get_spec()
    coremltools.utils.rename_feature(spec, current_name='input.1', new_name='input_tensor')
    coremltools.utils.rename_feature(spec, current_name='12', new_name='network_output')
    mlmodel = coremltools.models.MLModel(spec)

    # Save converted CoreML model
    mlmodel.save(model_out)


def generate_input_output_names(model_name):
    if model_name == "super_resolution":
        return ['input'], ['output']
    elif model_name == 'resnet18':
        return ['input'], ['output']
    else: 
        return ['inputs','labels'], ['outputs']


def main(model_name):
    onnx_path, coreml_path = onnx_coreml_paths(model_name)
    input_names, output_names = generate_input_output_names(model_name)
    converter_1(onnx_path, coreml_path, input_names, output_names)
    #converter_2(onnx_path, coreml_path)


if __name__ == "__main__":
    # command format  python onnx_to_coreml.py <model_name>
    parser = argparse.ArgumentParser(description="converts onnx model to coreml.")
    parser.add_argument("model_name", help="name of the model.")
    args = parser.parse_args()

    main(args.model_name)
