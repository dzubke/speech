# standard libraries
import argparse

# third party libraries
import coremltools
import onnx
import onnxruntime

#project libraries
from get_paths import onnx_coreml_paths 
from get_test_input import generate_test_input

# Load the model
#model = coremltools.models.MLModel('HousePricer.mlmodel')

# Make predictions
#predictions = model.predict({'bedroom': 1.0, 'bath': 1.0, 'size': 1240})

def validate_coreml(model_name):
    """
    """

    onnx_path, coreml_path = onnx_coreml_paths(model_name)
    test_input = generate_test_input("onnx", model_name)
    onnx_out = onnx_output(test_input, onnx_path)


def onnx_output(test_input, onnx_path):
    onnx_model = onnx.load(onnx_path)
    print(f"onnx model type: {type(onnx_model)}")
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_path)

    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_outputs = ort_session.run(None, ort_inputs)
    print(f"ort_inputs: {ort_inputs}, ort_outputs: {ort_outputs}")

def main(model_name):
    validate_coreml(model_name)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="validates a coreml model against an onnx model.")
    parser.add_argument("model_name", help="the name of the model")
    args = parser.parse_args()

    main(args.model_name)


    
