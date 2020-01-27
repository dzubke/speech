import torchvision.models as models
from torch import save

def model_generator():
    """this method will generate and save a native pytorch model to be converted by the converters
    """
    model = models.resnet18(pretrained=True)
    model_name = "resnet18"
    

    save(model, "./torch_models/"+model_name+".pth")


if __name__ == "__main__":
    model_generator()
