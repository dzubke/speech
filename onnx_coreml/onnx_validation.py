#some standard imports
import io
import argparse
import numpy as np

import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import onnx
import onnxruntime

from get_paths import pytorch_onnx_paths
from get_test_input import generate_test_input

# Super Resolution model definition in PyTorch
import torch.nn as nn
import torch.nn.init as init


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

def pytorch_tutorial_onnxruntime():
    """this block of code along with the SuperResolutionNet class were taken from the pytorch tutorial:
        https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    """
    # Create the super-resolution model by using the above model definition.
    torch_model = SuperResolutionNet(upscale_factor=3)

    #---------------------------------------

    # Load pretrained model weights
    model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
    batch_size = 1    # just a random number

    # Initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

    # set the model to inference mode
    torch_model.eval()


    #------------------------------------------
    # Input to the model
    x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
    torch_out = torch_model(x)
    print(f"torch_out :{torch_out}")
    
    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                    'output' : {0 : 'batch_size'}})

    torch.save(torch_model, "./pytorch_models/super_resolution.pth")

    #------------------------------------------

    import onnx
    import onnxruntime
    onnx_model = onnx.load("super_resolution.onnx")
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession("super_resolution.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    #----------------------
    from PIL import Image
    import torchvision.transforms as transforms

    img = Image.open("./cat_224x224.jpg")

    resize = transforms.Resize([224, 224])
    img = resize(img)

    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()

    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y)
    img_y.unsqueeze_(0)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

    # get the output image follow post-processing step from PyTorch implementation
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]).convert("RGB")

    # Save the image, we will compare this with the output image from mobile device
    final_img.save("./cat_superres_with_ort.jpg")


def test_onnxruntime(model_name):
    """this function takes in a pytorch_model as input
    """    
    
    input_tensor = generate_test_input("pytorch", model_name)
    print(f"input_tensor type: {type(input_tensor)}")
    torch_path, onnx_path = pytorch_onnx_paths(model_name)
    print(f"torch_path: {torch_path}") 
    torch_output = torch_export_inference(torch_path, onnx_path, input_tensor)
    
    onnx_runtime(input_tensor, onnx_path, torch_output)

def torch_export_inference(torch_path: str, onnx_path:str, input_tensor):
    """takes in a path to a pytorch model, loads the model, conducts inference on the input_tensor
        and exports the pytorch model as an onnx model. the method outputs the torch_ouput tensor that contains
        the inference
    """
    

    torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    torch_model = torch.load(torch_path, map_location=torch.device(torch_device))
    print(f"torch_model: {type(torch_model)}")


    # set the model to inference mode
    torch_model.eval()


    #------------------------------------------
    # Input to the model
    torch_output = torch_model(input_tensor)
    print(f"torch_output: {torch_output}")
    
    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      input_tensor,              # model input (or a tuple for multiple inputs)
                      onnx_path,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=9,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size', 1: 'time_dim'},    # variable lenght axes
                                    'output' : {0 : 'batch_size', 1: 'time_dim'}})
    
    return torch_output


def to_numpy(tensor):
    if isinstance(tensor, list):
        tensor_list = []
        for _tensor in tensor:
            numpy_tensor = _tensor.detach().cpu().numpy() if _tensor.requires_grad else _tensor.cpu().numpy()
            tensor_list.append(numpy_tensor)
        return tensor_list

    else:
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def onnx_runtime(input_tensor, onnx_path, torch_output):

    onnx_model = onnx.load(onnx_path)
    print(f"onnx model type: {type(onnx_model)}")
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_path)
    
    # compute ONNX Runtime output prediction
    print(f"ort_session.get_inputs(): {ort_session.get_inputs()}")
    if isinstance(input_tensor, list):
        ort_inputs = {"inputs": to_numpy(input_tensor[0]), "labels": to_numpy(input_tensor[1])}
    else:
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(f"{ort_outs}, {np.shape(ort_outs)}")

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_output), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def main(model_name):
    #convert_pytorch_onnx()
    test_onnxruntime(model_name)



if __name__ == "__main__":
    # commmand format: python onnx_runtime.py <model_name>
    parser = argparse.ArgumentParser(description="paths to test onnxruntime.")
    parser.add_argument("model_name", help="name of the model.")
    args = parser.parse_args()

    main(args.model_name)    
