import torch


def check_nan(model):
    """
        this function checks if a torch model includes any zero, nan, or infinite values 
        in its gradients
    """

    model