# standard libraries
from typing import Tuple, Iterable, List
# third-party libraries
import numpy as np
import torch

# used to specify a range of values within which a single value will be uniformly selected
# the selected value will be input to an augmentation function
AugmentRange = Tuple[float, float]

# output from torch.model.named_parameters()
TorchNamedParams = Iterable[Tuple[str, torch.nn.parameter.Parameter]]

# output of torch.model.parameters()
TorchParams = Iterable[torch.nn.parameter.Parameter]

# batches from the dataloader
Batch = Tuple[Tuple[np.ndarray], Tuple[List[str]]]
