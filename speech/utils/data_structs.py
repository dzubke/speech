from typing import Tuple, Iterable
import torch

# used to specify a range of values within which a single value will be uniformly selected
# the selected value will be input to an augmentation function
AugmentRange = Tuple[float, float]

# output from torch.model.named_parameters()
NamedParams = Iterable[Tuple[str, torch.nn.parameter.Parameter]]
