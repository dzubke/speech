import argparse

import torchvision.models as models
from torch import save, cuda

from get_paths import pytorch_onnx_paths

def model_generator(model_name):
    """this method will generate and save a native pytorch model to be converted by the converters.
        native pytroch model names can be found here: https://pytorch.org/docs/stable/torchvision/models.html
    """
    
    device = ".cuda()" if cuda.is_available() else ""
    model = eval("models."+model_name+"(pretrained=True)"+device)
    model_path, _ = pytorch_onnx_paths(model_name)
    
    save(model, model_path)


if __name__ == "__main__":
    # commmand format: python model_generator.py <model_name>
    parser = argparse.ArgumentParser(description="saves a native pytorch model")
    parser.add_argument("model_name", help="name of the model.")
    args = parser.parse_args()

    model_generator(args.model_name)
