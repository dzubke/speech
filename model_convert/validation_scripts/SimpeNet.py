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

        # in_ch 1, out_ch 1, kernel size (5,8), stride 1
        self.conv = nn.Conv2d(1, 3, (5, 8), stride=(1, 1), padding=0)
        # H_out: (10 - 5 + 1)/1 = 6 (time)      time gets crunched from 10 to 6
        # W_out: (11 - 8 + 1)/1 = 4 (freq)      freq gets crunched from 11 to 4

        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        self.h0 = torch.randn(1, 1, 3)
        
        # GRU params: input_size, hidden_size, num_layers
        # input_size = conv_out_ch * conv_w_out
        self.gru = nn.GRU(12, 3, num_layers=1, batch_first=True)

    def forward(self, x, h_prev):
        # to support streaming, this x input is actually static - it's a fixed-size window of audio

        # Initial input is (batch, time, freq)
        # Unsqueeze to add single channels dimension at index 1
        x = x.unsqueeze(1)

        # -------------

        # Conv2D
        # Input: (batch, channels, H=time, W=freq)
        # print("shape before conv: ", x.size()) # [1, 1, 100, 161]
        x = self.conv(x)
        
        # Output: (batch, channels, time, freq)
        # print("shape after conv: ", x.size()) # [1, 3, 96, 130]

        # Reshape for GRU
        # Transpose to (batch, time, channels, freq)
        x = torch.transpose(x, 1, 2).contiguous()
        # print("shape after transpose: ", x.size()) # [1, 96, 3, 130]
        
        x = flatten_orig(x)
        # x = flatten_scripted(x)
        # x = flatten_dustin(x)

        # print("shape after flattening: ", x.size()) # [1, 96, 390]

        # GRU (batch_first=True)
        # Input 1: (batch, seq, feature input)
        # Input 2 optional: hidden initial state (num_layers * num_directions, batch, hidden_size)
        x, h = self.gru(x, h_prev)
        
        # Output: (batch, seq, hidden size) = (1, seq, 20)
        # print("x after gru: ", x.size()) # [1, 96, 20]

        # (batch, 96, hidden_size)
        return x, h

def flatten_orig(x):
    # Flatten freq*channels to single feature dimension
    # Note: CoreML conversion will break if we do `x.size()` here instead of `x.data.size()`. See:
    # https://github.com/pytorch/pytorch/issues/11670#issuecomment-452697486
    b, t, c, f = x.data.size()
    x = x.view((b, t, c * f))
    return x

# Testing the Reshape node (for GRU input) using ONNX scripting instead of tracing:
# https://pytorch.org/docs/stable/onnx.html#tracing-vs-scripting
@torch.jit.script
def flatten_scripted(x):
    b, t, f, c = x.size()
    x = x.view((b, t, f * c))
    return x

# Dustin replacement for flatten
def flatten_dustin(x):
    # Say the output of the conv layer post-tranpose is (b, t, ch, freq) = (1, 196, 3, 130). 
    # Split tensor into 3 different tensors of shape (1, 196, 1, 130). 
    x = torch.split(x, 1, dim=2)
    # Concatenate into tensor of shape (1, 196, 1, 390) (390 = 3*130).
    x = torch.cat(x, dim=3)
    # Squeeze removes the single dimension in the middle to get (1, 196, 390).
    x = x.squeeze(2)
    return x

def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():
    print(f"torch version: {torch.__version__}")
    print(f"onnx version: {onnx.__version__}")
    print(f"onnx_coreml version: {onnx_coreml.__version__}")
    print(f"onnxruntime version: {onnxruntime.__version__}")
    print(f"coremltools version: {coremltools.__version__}")
    
    # Run Torch model once
    model = SimpleNet()
    model.eval()

    # Export Torch to ONNX with dummy input
    dummy_x = torch.zeros(1, 10, 11) # dummy input to model, shape (batch, time, freq)
    dummy_h0 = torch.zeros((1, 1, 3)) # h_0 of shape (num_layers * num_directions, batch, hidden_size):
    dummy_input = (dummy_x, dummy_h0)
    current_date = date.today()

    onnx_filename = f"SimpleNet_{current_date}.onnx"

    torch.onnx.export(model, dummy_input, onnx_filename,
                    # export_params=True,
                    verbose=True, 
                    input_names=['input', 'h_prev'],
                    output_names=['output', 'hidden'],
                    opset_version=9
                    # dynamic_axes={'input': {1: 'time_dim'}, 'output': {1: 'time_dim'}},
                    # do_constant_folding=True
                    )
    
    # Load and check new ONNX model
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    
    print("----- ONNX printable graph -----")
    # print(onnx_model.graph.value_info)
    print("~~~")
    print(onnx_model.graph.input)
    print(onnx_model.graph.output)
    print(onnx.helper.printable_graph(onnx_model.graph))
    print("-------------------")

    inferred_model = shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(inferred_model)
    # print(inferred_model.graph.value_info)
    print("~~~")

    # Testing ONNX output against Torch model
    test_x = torch.zeros(1, 10, 11)
    test_h0 = torch.zeros((1, 1, 3))
    test_input = (test_x, test_h0)

    # Predict with Torch
    torch_output, torch_h = model(test_x, test_h0)
    print("--------\nTORCH TEST")
    print(f"Test output {np.shape(to_numpy(torch_output))}: \n{to_numpy(torch_output)}")
    print(f"Test hidden {np.shape(to_numpy(torch_h))}: \n{to_numpy(torch_h)}")

    # Predict with ONNX using onnxruntime
    ort_session = onnxruntime.InferenceSession(onnx_filename)
    ort_inputs = {
        ort_session.get_inputs()[0].name: to_numpy(test_x),
        ort_session.get_inputs()[1].name: to_numpy(test_h0)
    }
    ort_out, ort_hidden = ort_session.run(None, ort_inputs)
    ort_np_out = np.array(ort_out)
    ort_np_hidden = np.array(ort_hidden)
    
    print("--------\nONNX TEST")
    print(f"Test output {np.shape(ort_np_out)}: \n{ort_np_out}")
    print(f"Test hidden {np.shape(ort_np_hidden)}: \n{ort_np_hidden}")
    
    # Compare Torch and ONNX predictions
    np.testing.assert_allclose(to_numpy(torch_output), ort_out, rtol=1e-03, atol=1e-05)
    print("Torch and ONNX predictions match, all good!")

    # Export to CoreML
    print("\n\nTrying to convert to CoreML...")
    mlmodel_filename = f"SimpleNet_{current_date}.mlmodel"
    mlmodel = onnx_coreml.convert(model=onnx_model, 
        # add_custom_layers=True,
        minimum_ios_deployment_target='13')
    mlmodel.save(mlmodel_filename)
    print(f"\nCoreML model saved to: {mlmodel_filename}\n")
    
    # Predict with CoreML
    spec = mlmodel.get_spec()

    from coremltools.models.neural_network.printer import print_network_spec
    # print_network_spec(spec, style='coding')
    # print_network_spec(spec)

    # print(f"CoreML Spec: {spec}\n\n")
    # coremltools.models.utils.save_spec(spec, 'saved_from_spec.mlmodel')
    # converted_mlmodel = coremltools.models.MLModel(spec)
    # print(f"Spec converted back to mlmodel and saved again")

    # print("Visualizing spec...")
    # mlmodel.visualize_spec()

    # Predict CoreML
    predict_input = {'input': to_numpy(test_x), 'h_prev': to_numpy(test_h0)}
    predictions = mlmodel.predict(predict_input)
    print(f"Prediction output: \n{predictions['output']}")
    print(f"Prediction hidden: \n{predictions['hidden']}")

if __name__ == "__main__":
    main()
