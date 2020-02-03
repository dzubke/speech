import numpy as np

from datetime import date
import torch
import torch.nn as nn
import onnx
from onnx import onnx_pb
import onnxruntime
from onnx_coreml import convert
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
        
        # Flatten freq*channels to single feature dimension
        # Note: CoreML conversion will break if we do `x.size()` here instead of `x.data.size()`. See:
        # https://github.com/pytorch/pytorch/issues/11670#issuecomment-452697486
        #b, t, f, c = x.data.size()
        #x = x.view((b, t, f * c))
        # x = flatten(x)
        x = torch.split(x, 1, dim=2)
        x = torch.cat(x, dim=3)
        x = x.squeeze(2)



        print("shape after flattening: ", x.size()) # [1, 196, 4160]

        # GRU (batch_first=True)
        # Input 1: (batch, seq, feature input)
        # Input 2 optional: hidden initial state (num_layers * num_directions, batch, hidden_size)
        x, h = self.gru(x)

        # Output: (batch, seq, hidden size) = (1, seq, 20)
        print("x after gru: ", x.size()) # [1, 196, 20]
        return x

# Testing the Reshape node (for GRU input) using ONNX scripting instead of tracing:
# https://pytorch.org/docs/stable/onnx.html#tracing-vs-scripting
@torch.jit.script
def flatten(x):
    b, t, f, c = x.size()
    x = x.view((b, t, f * c))
    return x

def main():
    # Run Torch model once
    model = SimpleNet()
    model.eval()
    
    # Export Torch to ONNX with dummy input
    dummy_input = torch.randn(1, 200, 161) # dummy input to model, shape (batch, time, freq)
    current_date = date.today()

    torch.save(model, f"./torch_models/SimpleNet_{current_date}.pth")
    onnx_filename = f"./onnx_models/SimpleNet_{current_date}.onnx"
    torch.onnx.export(model, dummy_input, onnx_filename,
                    export_params=True,
                    verbose=True, 
                    opset_version=9,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0:'batch_size', 1: 'time_dim'}, 'output': {0:'batch_size', 1: 'time_dim'}}
                    )
    
    # Load and check new ONNX model
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    
    print("----- ONNX printable graph -----")
    print(onnx_model.graph.input)
    print(onnx_model.graph.output)
    print(onnx.helper.printable_graph(onnx_model.graph))
    print("-------------------")

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
    mlmodel_filename = f"./coreml_models/SimpleNet_{current_date}.mlmodel"
    mlmodel = convert(model=onnx_model, minimum_ios_deployment_target='13')
    mlmodel.save(mlmodel_filename)
    mlmodel_data = {'data':to_numpy(test_input)}
    predictions = mlmodel.predict(mlmodel_data)    
    print(f"predictions: {predictions}")

if __name__ == "__main__":
    main()

