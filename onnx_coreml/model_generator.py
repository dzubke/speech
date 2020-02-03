import argparse

import torchvision.models as models
from torch import save, cuda
import torch.nn as nn
import torch.nn.init as init

from get_paths import pytorch_onnx_paths


class TestNet(nn.Module):
    def __init__(self, inplace=False):
        super(TestNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 32, (5, 32), 1, 0)
        self._initialize_weights()

    def forward(self, x):
        #x = self.relu(self.conv1(x))
        x = self.conv1(x)
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))




def prebuilt_generator(model_name):
    """this method will generate and save a native pytorch model to be converted by the converters.
        native pytroch model names can be found here: https://pytorch.org/docs/stable/torchvision/models.html
    """
    
    device = ".cuda()" if cuda.is_available() else ""
    model = eval("models."+model_name+"(pretrained=True)"+device)
    
    return model

def main(model_name):
    
    prebuilt_models = ["resnet18", "alexnet", "vgg16", "squeezenet", "densenet", "inception" , "googlenet" ,
                        "shufflenet", "mobilenet", "resnext50_32x4d", "wide_resnet50_2", "mnasnet"]
    
    if model_name in prebuilt_models:
        moodel = prebuilt_generator(model_name)
    
    else:
        model = TestNet()

    model_path, _ = pytorch_onnx_paths(model_name)
    save(model, model_path)

if __name__ == "__main__":
    # commmand format: python model_generator.py <model_name>
    parser = argparse.ArgumentParser(description="saves a native pytorch model")
    parser.add_argument("model_name", help="name of the model.")
    args = parser.parse_args()

    main(args.model_name)
