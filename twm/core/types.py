'''Shared type aliases used across the package.'''

from typing import Dict, Tuple

import numpy as np
import torch

Array = np.ndarray
ArrayDict = Dict[str, Array]
Shape = Tuple[int, ...]
Tensor = torch.Tensor
TensorDict = Dict[str, Tensor]

__all__ = ['Array', 'ArrayDict', 'Shape', 'Tensor', 'TensorDict']
