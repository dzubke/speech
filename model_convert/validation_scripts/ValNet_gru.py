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

torch.manual_seed(0)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 3, (3, 3), stride=(1, 1), padding=0)

        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        self.h0 = torch.randn(1, 1, 3)
        
        # GRU params: input_size, hidden_size, num_layers
        #self.gru = nn.GRU(9, 3, num_layers=1, batch_first=True)

    def forward(self, x):#, h_prev):
        # Initial input is (batch, time, freq)
        x = x.unsqueeze(1)

        # Conv2D
        # Input: (batch, channels, H=time, W=freq)
        x = self.conv(x)
        # Output: (batch, channels, time, freq)

        # Reshape for GRU
        # Transpose to (batch, time, channels, freq)
        x = torch.transpose(x, 1, 2).contiguous()
        
        x = flatten_orig_data(x)
        # x = flatten_scripted(x)
        #x = flatten_dustin(x)


        # GRU (batch_first=True)
        # Input 1: (batch, seq, feature input)
        # Input 2 optional: hidden initial state (num_layers * num_directions, batch, hidden_size)
        #x, h = self.gru(x, h_prev)

        # Output: (batch, seq, hidden size) = (1, seq, 20)
        return x#, h

def flatten_orig(x):

    b, t, c, f = x.size()
    x = x.view((b, t,c * f))
    return x

def flatten_orig_data(x):

    b, t, c, f = x.data.size()
    x = x.view((b, t, c* f))
    return x

def flatten_dustin(x):
    x = torch.split(x, 1, dim=2)
    x = torch.cat(x, dim=3)
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
    
    onnx_base = "../onnx_models/"
    coreml_base = "../coreml_models/"

    # Create the torch model
    model = SimpleNet()
    model.eval()


    # Export Torch to ONNX \
    dummy_x = torch.randn(1, 5, 5)
    dummy_h = torch.randn(1, 1, 3)
    dummy_input = (dummy_x, dummy_h)
    current_date = date.today()

    onnx_filename = f"SimpleNet_{current_date}.onnx"
    onnx_path = os.path.join(onnx_base, onnx_filename)
    torch.onnx.export(model, dummy_x, onnx_path, export_params=True, verbose=True,
                    input_names=['input'],# 'hidden_in'],
                    output_names=['output'],# 'hidden_out'],
                    opset_version=10,
                    do_constant_folding=True,
                    strip_doc_string=False
                    #dynamic_axes={'input': {0: 'batch', 1: 'time_dim'}, 'output': {0: 'batch', 1: 'time_dim'}}
                    )


    # Load and check new ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    inferred_model = shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(inferred_model)


   # Export to CoreML
    print("Try to convert to CoreML...")
    mlmodel_filename = f"SimpleNet_{current_date}.mlmodel"
    mlmodel_path = os.path.join(coreml_base, mlmodel_filename)
    mlmodel = onnx_coreml.convert(model=onnx_path, minimum_ios_deployment_target='13')
    mlmodel.save(mlmodel_path)

    print(f"\nCoreML model saved to: {mlmodel_path}\n")


    print("----- ONNX printable graph -----")
    #print(onnx_model.graph.value_info)
    print("~~~")
    #print(onnx_model.graph.input)
    #print(onnx_model.graph.output)
    #print(onnx.helper.printable_graph(onnx_model.graph))
    print("-------------------")

    #print(inferred_model.graph.value_info)
    print("~~~")

    print("----- CoreML printable graph -----")
    print(f"model __repr__: {mlmodel}")
    #print(f"model spec: {mlmodel.get_spec()}")
    from coremltools.models.neural_network.printer import print_network_spec
    #print_network_spec(mlmodel.get_spec(), style='coding')
    #mlmodel.visualize_spec(input_shape_dict={'input':[1, 1, 1, 5, 5]})
    

    print("----- Model Validation -----")

    test_x = torch.randn(1, 5, 5) # different time_dim to test dynamic seq length
    test_x_np = to_numpy(test_x).astype(np.float32)

    test_h = torch.randn(1, 1, 3)
    test_h_np = to_numpy(test_h).astype(np.float32)

    print(f"test_x input {test_x_np.shape}: \n{test_x_np}\n")
    print(f"test_h input {test_h_np.shape}: \n{test_h_np}\n")

    
    # Predict with Torch
    torch_output = model(test_x)#, test_h) , torch_h

    # Predict with Onnx
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: test_x_np}#, ort_session.get_inputs()[1].name: test_h_np}
    ort_output = ort_session.run(None, ort_inputs) #, ort_h
    ort_output = ort_output[0]

    #Predict with Coreml
    coreml_output_dict = mlmodel.predict({'input':test_x_np}, useCPUOnly=True) #, 'hidden_in': test_h_np
    coreml_output  = coreml_output_dict['output']# , coreml_output_dict['hidden_out'] , coreml_h =


    print(f"torch test output {to_numpy(torch_output).shape}: \n{to_numpy(torch_output)}\n")
    print(f"onnx test output {np.shape(ort_output)}: \n{ort_output}\n")
    print(f"coreml test output {np.shape(coreml_output)}: \n{coreml_output}\n")

    #print(f"torch test hidden {to_numpy(torch_h).shape}: \n{to_numpy(torch_h)}\n")
    #print(f"onnx test hidden {np.shape(ort_h)}: \n{ort_h}\n")
    #print(f"coreml test hidden {np.shape(coreml_h)}: \n{coreml_h}\n")


    # Compare Torch and ONNX predictions
    np.testing.assert_allclose(to_numpy(torch_output), ort_output, rtol=1e-05, atol=0)
    print("Torch and ONNX output predictions match!")
    #np.testing.assert_allclose(to_numpy(torch_h), ort_h, rtol=1e-05, atol=0)
    #print("Torch and ONNX output hidden states match!\n")

    
    coreml_rtol = 1e-05
    coreml_atol = 0

    np.testing.assert_allclose(to_numpy(torch_output), coreml_output, rtol=coreml_rtol, atol=coreml_atol)
    print("Torch and Coreml output predictions match!")
    #np.testing.assert_allclose(to_numpy(torch_h), coreml_h, rtol=coreml_rtol, atol=coreml_atol)
    #print("Torch and Coreml hidden states match!\n")


    np.testing.assert_allclose(coreml_output, ort_output, rtol=coreml_rtol, atol=coreml_atol)
    print("ONNX and Coreml predictions match!")
    #np.testing.assert_allclose(coreml_h, ort_h, rtol=coreml_rtol, atol=coreml_atol)
    #print("Torch and Coreml hidden states match!\n")





if __name__ == "__main__":
    main()

