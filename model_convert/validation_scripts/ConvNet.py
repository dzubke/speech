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

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # in_ch 1, out_ch 1, kernel size (5,8), stride 1
        self.conv = nn.Conv2d(1, 3, (5, 8), stride=(1, 1), padding=0)
        # H_out: (10 - 5 + 1)/1 = 6 (time)      time gets crunched from 10 to 6
        # W_out: (11 - 8 + 1)/1 = 4 (freq)      freq gets crunched from 11 to 4

    def forward(self, x):
        # Conv2D
        # Input: (batch, ch_in, H=time, W=freq)
        x = self.conv(x)
        # Output: (batch, ch_out, H_out, W_out) = (1, 3, 6, 4)
        return x

def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():    
    # Run Torch model once
    model = Net()
    model.eval()

    # Export Torch to ONNX with dummy input
    dummy_input = torch.randn(1, 1, 10, 11) # dummy input to model, shape (batch, ch, time, freq)
    current_date = date.today()

    onnx_filename = f"./onnx_models/ConvNet_{current_date}.onnx"

    torch.onnx.export(model, dummy_input, onnx_filename,
                    # export_params=True,
                    verbose=True, 
                    input_names=['input'],
                    output_names=['output'],
                    opset_version=9,
                    do_constant_folding=True
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
    test_input = torch.zeros(1, 1, 10, 11)
    
    # Predict with Torch
    torch_output = model(test_input)
    print("--------\nTORCH TEST")
    print(f"Test output {np.shape(to_numpy(torch_output))}: \n{to_numpy(torch_output)}")
    
    # Predict with ONNX using onnxruntime
    ort_session = onnxruntime.InferenceSession(onnx_filename)
    ort_input = {
        ort_session.get_inputs()[0].name: to_numpy(test_input)
    }
    ort_out = ort_session.run(None, ort_input)
    ort_np_out = np.array(ort_out[0])
    
    print("--------\nONNX TEST")
    print(f"Test output {np.shape(ort_np_out)}: \n{ort_np_out}")
    
    # Compare Torch and ONNX predictions
    np.testing.assert_allclose(to_numpy(torch_output), ort_np_out, rtol=1e-03, atol=1e-05)
    print("Torch and ONNX predictions match, all good!")

    # Export to CoreML
    print("\n\nTrying to convert to CoreML...")
    mlmodel_filename = f"./coreml_models/ConvNet_{current_date}.mlmodel"
    mlmodel = onnx_coreml.convert(model=onnx_model, minimum_ios_deployment_target='13')
    mlmodel.save(mlmodel_filename)
    print(f"\nCoreML model saved to: {mlmodel_filename}\n")
    
    # Predict with CoreML
    spec = mlmodel.get_spec()

    from coremltools.models.neural_network.printer import print_network_spec
    print_network_spec(spec, style='coding')
    # print_network_spec(spec)

    # print(f"CoreML Spec: {spec}\n\n")
    # coremltools.models.utils.save_spec(spec, 'saved_from_spec.mlmodel')
    # converted_mlmodel = coremltools.models.MLModel(spec)
    # print(f"Spec converted back to mlmodel and saved again")

    # print("Visualizing spec...")
    #mlmodel.visualize_spec(input_shape_dict={'input':[1, 1, 1, 10, 11]})
    test_input_1 = torch.zeros(1, 1, 1, 10, 11)
    predictions = mlmodel.predict({'input':to_numpy(test_input_1)}, useCPUOnly=True)


if __name__ == "__main__":
    main()
