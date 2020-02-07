import math
from collections import OrderedDict
import numpy as np
import json
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

class CTC(nn.Module):
    def __init__(self, freq_dim, time_dim, output_dim, config):
        super().__init__()
        
        encoder_cfg = config["encoder"]
        conv_cfg = encoder_cfg["conv"]

        convs = []
        in_c = 1
        for out_c, h, w, s in conv_cfg:     
            conv = nn.Conv2d(in_channels=in_c, 
                             out_channels=out_c, 
                             kernel_size=(h, w),
                             stride=(s, s), 
                             padding=(0, 0))
            batch_norm =  nn.BatchNorm2d(out_c)
            convs.extend([conv, nn.ReLU()])
            if config["dropout"] != 0:
                convs.append(nn.Dropout(p=config["dropout"]))
            in_c = out_c

        self.conv = nn.Sequential(*convs)
        conv_out = out_c * self.conv_out_size(freq_dim, 1)
        
        assert conv_out > 0, \
          "Convolutional output frequency dimension is negative."

        print(f"conv_out: {conv_out}")
        rnn_cfg = encoder_cfg["rnn"]
        self.rnn = nn.GRU(input_size=conv_out,
                          hidden_size=rnn_cfg["dim"],
                          num_layers=rnn_cfg["layers"],
                          batch_first=True, dropout=config["dropout"],
                          bidirectional=rnn_cfg["bidirectional"])
        _encoder_dim = rnn_cfg["dim"]

        # include the blank token
        print(f"fc _encoder_dim {_encoder_dim}, output_dim {output_dim}")
        self.fc = LinearND(_encoder_dim, output_dim + 1)

    def conv_out_size(self, n, dim):
        for c in self.conv.children():
            if type(c) == nn.Conv2d:
                # assuming a valid convolution meaning no padding
                k = c.kernel_size[dim]
                s = c.stride[dim]
                p = c.padding[dim]
                n = (n - k + 1 + 2*p) / s
                n = int(math.ceil(n))
        return n

    def forward(self, x, h_prev):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.transpose(x, 1, 2).contiguous()
        b, t, f, c = x.data.size()
        x = x.view((b, t, f*c))
        
        x, h = self.rnn(x, h_prev)
        x = self.fc(x)
        x = torch.nn.functional.softmax(x, dim=2)
        return x, h


class LinearND(nn.Module):

    def __init__(self, *args):
        """
        A torch.nn.Linear layer modified to accept ND arrays.
        The function treats the last dimension of the input
        as the hidden dimension.
        """
        super(LinearND, self).__init__()
        self.fc = nn.Linear(*args)

    def forward(self, x):
        size = x.size()
        n = int(np.prod(size[:-1]))
        out = x.contiguous().view(n, size[-1])
        out = self.fc(out)
        size = list(size)
        size[-1] = out.size()[-1]
        return out.view(size)

def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():
    print(f"torch version: {torch.__version__}")
    print(f"onnx version: {onnx.__version__}")
    print(f"onnx_coreml version: {onnx_coreml.__version__}")
    print(f"onnxruntime version: {onnxruntime.__version__}")
    print(f"coremltools version: {coremltools.__version__}")
    
    # Run Torch model once
    with open('ctc_config_20200115.json', 'r') as fid:
        current_date = date.today()
        name = f"CTCNet_{current_date}"

        config = json.load(fid)
        model_cfg = config["model"]
        model = CTC(161, 100, 40, model_cfg)

        # torch.save(model.state_dict(), f"{name}.pth")

        # state_dict = torch.load(f"{name}.pth")
        state_dict = torch.load('state_dict_20200115-0120.pth')
        # print(state_dict)
        for k, v in state_dict.items():
            print(k, v.shape, v.dtype)
        model.load_state_dict(state_dict)
        print("loaded state dict successfully!")

        model.eval()
        process(model, name)

def process(torch_model, name):
    # Export Torch to ONNX with dummy input
    dummy_x = torch.randn(1, 100, 161) # dummy input to model, shape (batch, time, freq)
    dummy_h0 = torch.randn(4, 1, 256) # h_0 of shape (num_layers * num_directions, batch, hidden_size):
    dummy_input = (dummy_x, dummy_h0)
    
    onnx_filename = f"{name}.onnx"

    torch.onnx.export(torch_model, dummy_input, onnx_filename,
                    # export_params=True,
                    verbose=True, 
                    input_names=['input', 'h_prev'],
                    output_names=['output', 'hidden'],
                    opset_version=9,
                    # dynamic_axes={'input': {1: 'time_dim'}, 'output': {1: 'time_dim'}},
                    do_constant_folding=True
                    )
    
    # Load and check new ONNX model
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    
    print("----- ONNX printable graph -----")
    # print(onnx_model.graph.value_info)
    print("~~~")
    # print(onnx_model.graph.input)
    # print(onnx_model.graph.output)
    # print(onnx.helper.printable_graph(onnx_model.graph))
    print("-------------------")

    inferred_model = shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(inferred_model)
    # print(inferred_model.graph.value_info)
    print("~~~")



    # Export to CoreML
    print("\n\nTrying to convert to CoreML...")
    mlmodel_filename = f"{name}.mlmodel"
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



    # Testing ONNX output against Torch model
    test_x = torch.randn(1, 100, 161)
    test_h0 = torch.randn(4, 1, 256)
    test_input = (test_x, test_h0)

    # Predict with Torch
    torch_output, torch_h = torch_model(test_x, test_h0)
    torch_np_out = to_numpy(torch_output)
    torch_np_h = to_numpy(torch_h)
    
    print("\n----- Test Torch -----")
    print(f"output {np.shape(torch_np_out)}: \n{torch_np_out[0,0,:10]}")
    print(f"hidden {np.shape(torch_np_h)}: \n{torch_np_h[0,0,:10]}")

    # Predict with ONNX using onnxruntime
    ort_session = onnxruntime.InferenceSession(onnx_filename)
    ort_inputs = {
        ort_session.get_inputs()[0].name: to_numpy(test_x),
        ort_session.get_inputs()[1].name: to_numpy(test_h0)
    }
    ort_out, ort_hidden = ort_session.run(None, ort_inputs)
    ort_np_out = np.array(ort_out)
    ort_np_hidden = np.array(ort_hidden)
    
    print("\n----- Test ONNX -----")
    print(f"output {np.shape(ort_np_out)}: \n{ort_np_out[0,0,:10]}")
    print(f"hidden {np.shape(ort_np_hidden)}: \n{ort_np_hidden[0,0,:10]}")
    
    # Predict CoreML
    predict_input = {'input': to_numpy(test_x), 'h_prev': to_numpy(test_h0)}
    predictions = mlmodel.predict(predict_input, useCPUOnly=True)
    coreml_output = np.array(predictions['output'])
    coreml_hidden = np.array(predictions['hidden'])
    
    print("\n----- Test CoreML -----")
    print(f"output {np.shape(coreml_output)}: \n{coreml_output[0,0,:10]}")
    print(f"hidden {np.shape(coreml_hidden)}: \n{coreml_hidden[0,0,:10]}")

    # Compare Torch and ONNX predictions
    np.testing.assert_allclose(torch_np_out, ort_np_out, rtol=1e-03, atol=1e-05)
    print("\nTorch and ONNX outputs match, all good!")    

    np.testing.assert_allclose(torch_np_h, ort_np_hidden, rtol=1e-03, atol=1e-05)
    print("Torch and ONNX hidden match, all good!")    

    # Compare ONNX and CoreML predictions
    np.testing.assert_allclose(ort_np_out, coreml_output, rtol=5e-02, atol=5e-04)
    print("ONNX and CoreML outputs match, all good!")

    np.testing.assert_allclose(ort_np_hidden, coreml_hidden, rtol=5e-02, atol=5e-04)
    print("ONNX and CoreML hidden match, all good!")

if __name__ == "__main__":
    main()
