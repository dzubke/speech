import argparse
import os

from import_export import onnx_coreml_export



def main(model_name):
    onnx_path ="./onnx_models/"+model_name+"_model.onnx"
    coreml_path ="./coreml_models/"+model_name+"_model.mlmodel"


    onnx_coreml_export(onnx_path, coreml_path)
    #onnx_coreml_export_2(onnx_path, coreml_path)
    print(f"Onnx model successfully converted to CoreML at: {coreml_path}")

if __name__ == "__main__":
    # command format  python onnx_to_coreml.py <model_name>
    parser = argparse.ArgumentParser(description="converts onnx model to coreml.")
    parser.add_argument("model_name", help="name of the model.")
    args = parser.parse_args()

    main(args.model_name)
