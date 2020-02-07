import numpy as np

from datetime import date
import torch
import torch.nn as nn
import onnx
from onnx import onnx_pb
from onnx import helper, shape_inference

import onnxruntime
import onnx_coreml
import coremltools

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

        # in_ch 1, out_ch 1, kernel size (5,32), stride 1
        self.conv = nn.Conv2d(1, 32, (5, 32), stride=(1, 1), padding=0)
        # H_out: (200 - 5 + 1)/1 = 196 (time)
        # W_out: (161 - 32 + 1)/1 = 130 (freq)

        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        self.h0 = torch.randn(1, 1, 20)
        
        # GRU params: input_size, hidden_size, num_layers
        self.gru = nn.GRU(4160, 20, num_layers=1, batch_first=True)

    def forward(self, x):
        # Initial input is (batch, time, freq)
        # Unsqueeze to add single channels dimension at index 1
        x = x.unsqueeze(1)

        # Conv2D
        # Input: (batch, channels, H=time, W=freq)
        print("shape before conv: ", x.size()) # [1, 1, 200, 161]
        x = self.conv(x)
        # Output: (batch, channels, time, freq)
        print("shape after conv: ", x.size()) # [1, 32, 196, 130]

        # Reshape for GRU
        # Transpose to (batch, time, channels, freq)
        x = torch.transpose(x, 1, 2).contiguous()
        print("shape after transpose: ", x.size()) # [1, 196, 32, 130]
        
        # x = flatten_orig(x)
        # x = flatten_scripted(x)
        x = flatten_dustin(x)

        print("shape after flattening: ", x.size()) # [1, 196, 4160]

        # GRU (batch_first=True)
        # Input 1: (batch, seq, feature input)
        # Input 2 optional: hidden initial state (num_layers * num_directions, batch, hidden_size)
        x, h = self.gru(x)

        # Output: (batch, seq, hidden size) = (1, seq, 20)
        # print("x after gru: ", x.size()) # [1, 196, 20]
        return x

def flatten_orig(x):
    # Flatten freq*channels to single feature dimension
    # Note: CoreML conversion will break if we do `x.size()` here instead of `x.data.size()`. See:
    # https://github.com/pytorch/pytorch/issues/11670#issuecomment-452697486
    b, t, f, c = x.size()
    x = x.view((b, t, f * c))
    return x

# Testing the Reshape node (for GRU input) using ONNX scripting instead of tracing:
# https://pytorch.org/docs/stable/onnx.html#tracing-vs-scripting
#@torch.jit.script
#def flatten_scripted(x):
#    b, t, f, c = x.size()
#    x = x.view((b, t, f * c))
#    return x

# Dustin replacement for flatten
def flatten_dustin(x):
    # Say the output of the conv layer is (1, 196, 32, 130). 
    # Split tensor into 32 different tensors of shape (1, 196, 1, 130). 
    x = torch.split(x, 1, dim=2)
    # Concatenate into tensor of shape (1, 196, 1, 4160) (4160 = 32*130).
    x = torch.cat(x, dim=3)
    # Squeeze removes the single dimension in the middle to get (1, 196, 4160).
    x = x.squeeze(2)
    return x

def main():
    print(f"torch version: {torch.__version__}")
    print(f"onnx version: {onnx.__version__}")
    #print(f"onnx_coreml version: {onnx_coreml.__version__}")
    print(f"onnxruntime version: {onnxruntime.__version__}")
    #print(f"coremltools version: {coremltools.__version__}")
    
    # Run Torch model once
    model = SimpleNet()
    model.eval()

    # Export Torch to ONNX with dummy input
    dummy_input = torch.randn(1, 200, 161) # dummy input to model, shape (batch, time, freq)
    current_date = date.today()

    onnx_filename = f"SimpleNet_{current_date}.onnx"
    print("before export")
    torch.onnx.export(model, dummy_input, onnx_filename, export_params=True, verbose=True,
                    input_names=['input'],
                    output_names=['output'],
                    opset_version=9,
                    strip_doc_string=False,
                    dynamic_axes={'input': {0: 'batch', 1: 'time_dim'}, 'output': {0: 'batch', 1: 'time_dim'}}
                    )
    print("after export")

    # Load and check new ONNX model
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    
    print("----- ONNX printable graph -----")
    #print(onnx_model.graph.value_info)
    print("~~~")
    #print(onnx_model.graph.input)
    #print(onnx_model.graph.output)
    #print(onnx.helper.printable_graph(onnx_model.graph))
    print("-------------------")

    inferred_model = shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(inferred_model)
    #print(inferred_model.graph.value_info)
    print("~~~")

    # Testing ONNX output against Torch model
    test_input = torch.randn(1, 1000, 161) # different time_dim to test dynamic seq length
    
    # Predict with Torch
    torch_output = model(test_input)
    print(f"torch test output {torch_output.size()}: {torch_output}")

    # Predict with ONNX using onnxruntime
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    ort_session = onnxruntime.InferenceSession(onnx_filename)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(test_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(f"onnxruntime test output {np.shape(ort_outs)}: {ort_outs}")
    
    # Compare Torch and ONNX predictions
    np.testing.assert_allclose(to_numpy(torch_output), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Torch and ONNX predictions match, all good!")

    # Export to CoreML
    print("Try to convert to CoreML...")
    mlmodel_filename = f"SimpleNet_{current_date}.mlmodel"
    mlmodel = onnx_coreml.convert(model=onnx_model, minimum_ios_deployment_target='13')
    mlmodel.save(mlmodel_filename)
    print(f"\nCoreML model saved to: {mlmodel_filename}\n")
    
    # Predict with CoreML
    #mlmodel.visualize_spec()
    print(f"model __repr__: {mlmodel}")
    #mlmodel_data = to_numpy(test_input)
    mlmodel_data = {'data':to_numpy(test_input)}
    predictions = mlmodel.predict(mlmodel_data)
    print(f"mlmodel predictions: {predictions}")

    #print(f"model.get_spec: {mlmodel.get_spec()}")
    #print(f"model.visualize_spec: {mlmodel.visualize_spec(input_shape_dict={1:'batch', 200: 'time', 161:'freq'})}")




if __name__ == "__main__":
    main()
