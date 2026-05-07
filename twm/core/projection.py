import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

Tensor = torch.Tensor
TensorDict = Dict[str, Tensor]
Shape = Tuple[int, ...]


# <------------------------------- Vector In and Out  ------------------------------>

class VectorEncoder(nn.Module):
    '''An MLP to encode vector states and actions into a (d_model,) embedding.'''

    def __init__(self, input_shape: Shape, d_model: int) -> None:
        super().__init__()

        # simple MLP to project state and action vector into (d_model,) embedding
        in_dim = int(np.prod(input_shape, dtype=np.int64))
        n_hidden = (in_dim + d_model) // 2
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, d_model)
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len = x.shape[:2]
        x = x.view(batch * seq_len, -1)
        enc = self.encoder(x)
        enc = enc.view(batch, seq_len, -1)
        return enc


class VectorDecoder(nn.Module):
    '''An MLP to decode a (d_model,) embedding into vector states.'''
    condition_mode = 'last'

    def __init__(self, output_shape: Shape, d_model: int) -> None:
        super().__init__()
        self.output_shape = output_shape
        self.loss_fn = nn.HuberLoss()
        
        # simple MLP to project (d_model,) embedding back into state dict
        out_dim = int(np.prod(output_shape, dtype=np.int64))
        n_hidden = (d_model + out_dim) // 2
        self.decoder = nn.Sequential(
            nn.Linear(d_model, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, out_dim),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 2, 'Decoder input should be (batch, d_model).'
        dec = self.decoder(x)
        dec = dec.view(-1, *self.output_shape)
        return dec

    def compute_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        '''Default vector regression loss for real-valued states.'''
        return self.loss_fn(pred, target)

    def decode_output(self, tensor: Tensor, **kwargs) -> Tensor:
        '''Default output decode is identity; model handles denormalization for reals.'''
        return tensor.float()


class CategoricalDecoder(VectorDecoder):
    '''Decoder for int/bool states that owns CE loss and sampling logic.'''

    def __init__(self, output_shape: Shape, d_model: int, low: int, high: int) -> None:
        super().__init__(output_shape, d_model)
        self.low = low
        self.high = high
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = pred.movedim(-1, 1)
        target = target.movedim(-1, 1)
        return self.loss_fn(pred, target)

    def decode_output(self, tensor: Tensor, **kwargs) -> Tensor:
        grad = kwargs.get('grad', False)
        stochastic = kwargs.get('stochastic', False)
        temperature = kwargs.get('temperature', 1.0)

        # gradient required: output is softmax probabilities over possible values
        if grad:
            return torch.softmax(tensor.float(), dim=-1)
    
        # no gradient: output is sampled or argmax indices of the predicted distribution
        elif stochastic:
            probs = torch.softmax(tensor.float() / temperature, dim=-1)
            flat_probs = probs.reshape(-1, probs.shape[-1])
            idx = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
            idx = idx.view(*probs.shape[:-1])
        else:
            idx = tensor.argmax(dim=-1)
            
        assert idx.max() < (self.high - self.low + 1)
        return idx + self.low


# <------------------------------- Image In and Out  ------------------------------>

class ImageEncoder(nn.Module):
    '''A CNN to encode images into a (d_model,) embedding.'''

    def __init__(self, image_shape: Shape, d_model: int) -> None:
        super().__init__()
        
        if len(image_shape) != 3:
            raise ValueError('Image_shape should be (C, H, W).')
        c, h, w = image_shape

        self.proj = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, c, h, w = x.shape
        x = x.view(batch * seq_len, c, h, w)
        enc = self.proj(x)
        enc = enc.view(batch, seq_len, -1)
        return enc


class ImageDecoder(nn.Module):
    '''A CNN to decode a (d_model,) embedding into images.'''
    condition_mode = 'last'

    def __init__(self, image_shape: Shape, d_model: int) -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

        if len(image_shape) != 3:
            raise ValueError('Image_shape should be (C, H, W).')
        c, h, w = image_shape

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, 128 * 8 * 8),
            nn.GELU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Upsample(size=(h, w), mode='bilinear', align_corners=False),
            nn.Conv2d(32, c, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 2, 'Decoder input should be (batch, d_model).'
        return self.proj(x)

    def compute_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.loss_fn(pred, target)

    def decode_output(self, tensor: Tensor, **kwargs) -> Tensor:
        return torch.sigmoid(tensor).float()
    