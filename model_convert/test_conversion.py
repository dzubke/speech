import torch
import torch.nn as nn
import onnx
from onnx import onnx_pb
from onnx_coreml import convert
import coremltools



class TestNet(nn.Module):
    def __init__(self, inplace=False):
        super(TestNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 32, (5, 32), 1, 0)
        self.gru = nn.GRU(4160, 256, num_layers=1, batch_first=True)
    
    def forward(self, x):
        #x = self.relu(self.conv1(x))
        x = self.conv1(x)
        x = torch.transpose(x, 1, 2).contiguous()
        b, t, f, c = x.data.size()
        x = x.view((b, t, f * c))  
        x, h = self.gru(x)
        return x




layer_count=1
# model = nn.LSTM(10, 20, num_layers=layer_count, bidirectional=True)
#model = nn.Sequential(nn.Conv1d(1, 32, 5, 1, 0), nn.GRU(10, 20, num_layers=layer_count))
#model = nn.Conv2d(1, 32, (5, 32), 1, 0)
model = TestNet()
model.eval()
model_name = "20200130_1conv2d-1gru"

torch.save(model, './torch_models/'+model_name+'.pth')

with torch.no_grad():
    #input = torch.randn(5, 3, 10)
    #h0 = torch.randn(layer_count, 3, 20)
    #output, hn = model(input, h0)

    input = torch.rand(1, 1, 200, 161)


    # default export
    #torch.onnx.export(model, (input, h0), './onnx_models/'+model_name+'.onnx')
    torch.onnx.export(model, input, './onnx_models/'+model_name+'.onnx')

    onnx_model = onnx.load('./onnx_models/'+model_name+'.onnx')
    # input shape [5, 3, 10]
    print(onnx_model.graph.input[0])

    # export with `dynamic_axes`
    #torch.onnx.export(model, input, model_name+'.onnx',
    #                input_names=['input', 'h0'],
    #                output_names=['output', 'hn'],
    #                dynamic_axes={'input': {0: 'sequence'}, 'output': {0: 'sequence'}})
    
    #torch.onnx.export(model, input, model_name+'.onnx',
    #                input_names=['input'],
    #                output_names=['output']
    #                )

    onnx_model = onnx.load('./onnx_models/'+model_name+'.onnx')
    # input shape ['sequence', 3, 10]
    print(onnx_model.graph.input[0])

    mlmodel = convert(model=onnx_model, minimum_ios_deployment_target='13')
    mlmodel.save('./coreml_models/'+model_name+'.mlmodel')
