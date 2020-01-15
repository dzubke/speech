import torch

model = torch.load('best_model', map_location=torch.device('cpu'))

dummy_input = [torch.FloatTensor(1, 200, 161), torch.IntTensor(1, 41)]
torch.onnx.export(model, dummy_input, 'test.proto', verbose=True)
