import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime


layer_count = 4

model = nn.LSTM(10, 20, num_layers=layer_count, bidirectional=False, batch_first=True)
model.eval()
torch.save(model, "./torch_models/lstm.pth")

with torch.no_grad():
    input = torch.randn(5, 3, 10)
    h0 = torch.randn(layer_count * 2, 3, 20)
    c0 = torch.randn(layer_count * 2, 3, 20)
    torch_output, (hn, cn) = model(input, (h0, c0))

    # default export
    torch.onnx.export(model, (input, (h0, c0)), 'lstm.onnx')
    onnx_model = onnx.load('lstm.onnx')
    # input shape [5, 3, 10]
    print(onnx_model.graph.input[0])

    # export with `dynamic_axes`
    torch.onnx.export(model, (input, (h0, c0)), './onnx_models/lstm.onnx',
                    input_names=['input', 'h0', 'c0'],
                    output_names=['output', 'hn', 'cn'],
                    dynamic_axes={'input': {0: 'sequence'}, 'output': {0: 'sequence'}})
    onnx_filename = './onnx_models/lstm.onnx'
    onnx_model = onnx.load(onnx_filename)
    # input shape ['sequence', 3, 10]
    print(onnx_model.graph.input[0])
    onnx.checker.check_model(onnx_model)

    # Predict with ONNX using onnxruntime
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    test_input = torch.randn(5, 3, 10) # different time_dim to test dynamic seq length

    ort_session = onnxruntime.InferenceSession(onnx_filename)
    ort_inputs = {'input': to_numpy(input), 'h0': to_numpy(h0), 'c0': to_numpy(c0)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(f"onnxruntime test output {np.shape(ort_outs)}: {ort_outs}")

    # Compare Torch and ONNX predictions
    np.testing.assert_allclose(to_numpy(torch_output), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Torch and ONNX predictions match, all good!")
