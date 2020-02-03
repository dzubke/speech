import os

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
        self.conv = nn.Conv2d(1, 3, (3, 3), stride=(1, 1), padding=0)
        # H_out: (200 - 5 + 1)/1 = 196 (time)
        # W_out: (161 - 32 + 1)/1 = 130 (freq)

        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        #self.h0 = torch.randn(1, 1, 20)
        
        # GRU params: input_size, hidden_size, num_layers
        #self.gru = nn.GRU(9, 3, num_layers=1, batch_first=True)

    def forward(self, x):#, h_prev):
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
        
        #x = flatten_orig_data(x)
        # x = flatten_scripted(x)
        #x = flatten_dustin(x)

        print("shape after flattening: ", x.size()) # [1, 196, 4160]

        # GRU (batch_first=True)
        # Input 1: (batch, seq, feature input)
        # Input 2 optional: hidden initial state (num_layers * num_directions, batch, hidden_size)
        #x, h = self.gru(x, h_prev)

        # Output: (batch, seq, hidden size) = (1, seq, 20)
        # print("x after gru: ", x.size()) # [1, 196, 20]
        return x#, h

def flatten_orig(x):
    # Flatten freq*channels to single feature dimension
    # Note: CoreML conversion will break if we do `x.size()` here instead of `x.data.size()`. See:
    # https://github.com/pytorch/pytorch/issues/11670#issuecomment-452697486
    b, t, c, f = x.size()
    x = x.view((b, t,c * f))
    return x

def flatten_orig_data(x):
    # Flatten freq*channels to single feature dimension
    # Note: CoreML conversion will break if we do `x.size()` here instead of `x.data.size()`. See:
    # https://github.com/pytorch/pytorch/issues/11670#issuecomment-452697486
    b, t, c, f = x.data.size()
    x = x.view((b, t, c* f))
    return x

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
    
    onnx_base = "./onnx_models/"
    coreml_base = "./coreml_models/"

    # Run Torch model once
    model = SimpleNet()
    model.eval()

    # Export Torch to ONNX with dummy input
    dummy_x = torch.randn(1, 5, 5) # dummy input to model, shape (batch, time, freq)
    dummy_h = torch.randn(1, 1, 3)
    dummy_input = (dummy_x, dummy_h)
    current_date = date.today()

    onnx_filename = f"SimpleNet_{current_date}.onnx"
    onnx_path = os.path.join(onnx_base, onnx_filename)
    print("before export")
    torch.onnx.export(model, dummy_x, onnx_path, export_params=True, verbose=True,
                    input_names=['input'],# 'hidden_in'],
                    output_names=['output'],# 'hidden_out'],
                    opset_version=10,
                    do_constant_folding=True,
                    strip_doc_string=False
                    #dynamic_axes={'input': {0: 'batch', 1: 'time_dim'}, 'output': {0: 'batch', 1: 'time_dim'}}
                    )
    print("after export")

    # Load and check new ONNX model
    onnx_model = onnx.load(onnx_path)
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

    # Predict with ONNX using onnxruntime
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    # Testing ONNX output against Torch model
    test_x = torch.randn(1, 5, 5) # different time_dim to test dynamic seq length
    test_h = torch.randn(1, 1, 3)
    test_input = (test_x, test_h)
    print(f"test_x output {to_numpy(test_x).shape}: {to_numpy(test_x)}")
    print(f"test_h output {to_numpy(test_h).shape}: {to_numpy(test_h)}")

    
    # Predict with Torch
    torch_output = model(test_x)#, test_h) , torch_h
    print(f"torch test output {to_numpy(torch_output).shape}: {to_numpy(torch_output)}")
    
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(test_x)}#, ort_session.get_inputs()[1].name: to_numpy(test_h)}
    ort_outs = ort_session.run(None, ort_inputs) #, ort_h
    print(f"onnxruntime test output {np.shape(ort_outs)}: {ort_outs}")
    
    # Compare Torch and ONNX predictions
    np.testing.assert_allclose(to_numpy(torch_output), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Torch and ONNX predictions match, all good!")
    
    # Export to CoreML
    print("Try to convert to CoreML...")
    mlmodel_filename = f"SimpleNet_{current_date}.mlmodel"
    mlmodel_path = os.path.join(coreml_base, mlmodel_filename)
    mlmodel = onnx_coreml.convert(model=onnx_path, minimum_ios_deployment_target='13')
    print("before save/  after convert")
    mlmodel.save(mlmodel_path)
    print(f"\nCoreML model saved to: {mlmodel_path}\n")
    
    # Predict with CoreML
    #mlmodel.visualize_spec()
    print(f"model __repr__: {mlmodel}")
    print(f"model.get_spec: {mlmodel.get_spec()}")

    #mlmodel_data = to_numpy(test_input)
    print("before input")
    mlmodel_data = {'input':to_numpy(test_x).astype(np.float32)}#, 'hidden_in': to_numpy(test_h).tolist()}
    input_dict = {'input_ids': test_x.numpy().astype(np.float32)}
    print("after input/ before predict")
    #predictions = mlmodel.predict(mlmodel_data)
    predictions = mlmodel.predict(input_dict, useCPUOnly=True)


    print("after predict")

    #print(f"mlmodel predictions: {predictions}")

    #print(f"model.visualize_spec: {mlmodel.visualize_spec()}")#input_shape_dict={'batch':0, 'time':0, 'freq':161})}")




if __name__ == "__main__":
    main()

    """
    Scratch work

    # Testing the Reshape node (for GRU input) using ONNX scripting instead of tracing:
    # https://pytorch.org/docs/stable/onnx.html#tracing-vs-scripting
    #@torch.jit.script
    #def flatten_scripted(x):
    #    b, t, f, c = x.size()
    #    x = x.view((b, t, f * c))
    #    return x
    """