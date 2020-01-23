import sys
import argparse

from onnx import onnx_pb
from onnx_coreml import convert
import coremltools

def converter_1(model_in, model_out):
    model_file = open(model_in, 'rb')
    model_proto = onnx_pb.ModelProto()
    model_proto.ParseFromString(model_file.read())
    coreml_model = convert(
        model_proto,  image_input_names=['inputs, labels'], image_output_names=['179'], 
        minimum_ios_deployment_target = '13' 
    )
    coreml_model.save(model_out)

def converter_2(model_in, model_out):
    mlmodel = convert(model=model_in, minimum_ios_deployment_target='13')
      
    # rename inputs/outputs
    spec = mlmodel.get_spec()
    coremltools.utils.rename_feature(spec, current_name='input.1', new_name='input_tensor')
    coremltools.utils.rename_feature(spec, current_name='12', new_name='network_output')
    mlmodel = coremltools.models.MLModel(spec)

    # Save converted CoreML model
    mlmodel.save(model_out)

def main(model_in, model_out):
    
    #converter_1(model_in, model_out)
    converter_2(model_in, model_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Convert an onnx file to coreml.")
    parser.add_argument("model_in",
        help="The path to the input onnxe file that ends with onnx_model.onnx")
    parser.add_argument("model_out",
        help="The path of the output coreml file that ends with coreml_model.mlmodel")
    
    args = parser.parse_args()

    main(args.model_in, args.model_out)
