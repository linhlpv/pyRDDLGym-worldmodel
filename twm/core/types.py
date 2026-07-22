'''Shared type aliases used across the package.'''

from typing import Dict

import numpy as np
import torch

Array = np.ndarray
ArrayDict = Dict[str, Array]
Tensor = torch.Tensor
TensorDict = Dict[str, Tensor]

__all__ = ['Array', 'ArrayDict', 'Tensor', 'TensorDict']
