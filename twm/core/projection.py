import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from twm.core.types import Shape, Tensor, TensorDict

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


# <------------------------------- Image In and Out  -------------------------------->

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
    
    
# <--------------------------- Normalizing Flow Decoder -------------------------->

class _AffineCouplingLayer(nn.Module):
    '''RealNVP affine coupling.'''

    def __init__(self, dim: int, d_cond: int, hidden: int) -> None:
        super().__init__()
        self.d1 = dim // 2
        self.d2 = dim - self.d1

        # outputs both scale and shift concatenated: (d2, d2)
        self.net = nn.Sequential(
            nn.Linear(self.d1 + d_cond, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.d2 * 2),
        )

    def forward(self, x: Tensor, cond: Tensor, inverse: bool=False) -> Tuple[Tensor, Tensor]:
        x1, x2 = x[..., :self.d1], x[..., self.d1:]
        st = self.net(torch.cat([x1, cond], dim=-1))
        s, t = st[..., :self.d2], st[..., self.d2:]
        s = torch.tanh(s)    # bound scale, avoids exp explosion

        if not inverse:
            y2 = x2 * torch.exp(s) + t
            log_det = s.sum(-1)
        else:
            y2 = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(-1)

        return torch.cat([x1, y2], dim=-1), log_det


class FlowDecoder(nn.Module):
    '''Normalizing flow decoder using RealNVP affine coupling layers.'''
    condition_mode = 'last'

    def __init__(self, output_shape: Shape, d_model: int,
                 n_layers: int=4, hidden: int=128) -> None:
        super().__init__()
        self.output_shape = output_shape
        self.dim = int(np.prod(output_shape, dtype=np.int64))

        # project transformer latent to a fixed-size conditioning vector
        self.cond_proj = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # coupling layers; alternate which half is transformed by flipping after each
        self.layers = nn.ModuleList([
            _AffineCouplingLayer(self.dim, hidden, hidden)
            for _ in range(n_layers)
        ])

    def forward(self, z: Tensor) -> Tensor:
        return z

    def _run(self, x: Tensor, cond: Tensor, inverse: bool) -> Tuple[Tensor, Tensor]:
        '''Run all coupling layers in forward or inverse order.'''
        log_det = torch.zeros(x.shape[0], device=x.device)

        if not inverse:
            for layer in self.layers:
                x, ld = layer(x, cond, inverse=False)
                log_det = log_det + ld
                x = torch.flip(x, dims=[-1])
        else:
            for layer in reversed(self.layers):
                x = torch.flip(x, dims=[-1])
                x, ld = layer(x, cond, inverse=True)
                log_det = log_det + ld

        return x, log_det

    def compute_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        '''Exact NLL: map target → noise via inverse flow, evaluate N(0,I) log-prob.'''
        cond = self.cond_proj(pred)
        x = target.float().reshape(pred.shape[0], -1)
        eps, log_det = self._run(x, cond, inverse=False)
        log_p = -0.5 * (eps.pow(2) + math.log(2 * math.pi)).sum(-1)
        nll = -(log_p + log_det)
        return nll.mean()

    def decode_output(self, tensor: Tensor, **kwargs) -> Tensor:
        '''Map noise → state via forward flow conditioned on z.'''
        stochastic = kwargs.get('stochastic', False)
        temperature = kwargs.get('temperature', 1.0)
        cond = self.cond_proj(tensor)
        batch = tensor.shape[0]

        if stochastic:
            eps = torch.randn(batch, self.dim, device=tensor.device) * temperature
        else:
            eps = torch.zeros(batch, self.dim, device=tensor.device)
            
        x, _ = self._run(eps, cond, inverse=True)
        return x.reshape(batch, *self.output_shape).float()


# <------------------------ Diffusion Decoder (DDPM + DDIM) ---------------------->

class _SinusoidalPosEmb(nn.Module):
    '''Sinusoidal timestep embedding.'''

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, 'dim must be even'
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class _ResidualBlock(nn.Module):
    '''ResNet block with additive conditioning injection.'''

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.GroupNorm(self._norm_groups(in_ch), in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.layer2 = nn.Sequential(
            nn.GroupNorm(self._norm_groups(out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    @staticmethod
    def _norm_groups(ch: int) -> int:
        '''Largest divisor of ch that is <= 8, for use with GroupNorm.'''
        for g in (8, 4, 2, 1):
            if ch % g == 0:
                return g
        return 1

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.layer1(x)
        h = h + self.cond_proj(cond)[:, :, None, None]
        h = self.layer2(h)
        return h + self.skip(x)


class GaussianDiffusion(nn.Module):
    '''Gaussian diffusion module adapted to decoder API.'''
    condition_mode = 'last'

    def __init__(self, output_shape: Shape, d_model: int, T: int=256) -> None:
        super().__init__()
        self.output_shape = output_shape
        self.n_timesteps = T
        self.clip_denoised = True

        betas = self.cosine_beta_schedule(self.n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        self.betas: Tensor
        self.alphas_cumprod: Tensor
        self.alphas_cumprod_prev: Tensor
        self.sqrt_alphas_cumprod: Tensor
        self.sqrt_one_minus_alphas_cumprod: Tensor
        self.sqrt_recip_alphas_cumprod: Tensor
        self.sqrt_recipm1_alphas_cumprod: Tensor
        self.posterior_variance: Tensor
        self.posterior_log_variance_clipped: Tensor
        self.posterior_mean_coef1: Tensor
        self.posterior_mean_coef2: Tensor
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer(
            'sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer(
            'sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer(
            'posterior_log_variance_clipped', 
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer(
            'posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
    
    @staticmethod
    def cosine_beta_schedule(timesteps: int, s: float=0.008) -> Tensor:
        '''Cosine beta schedule.'''
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.tensor(np.clip(betas, a_min=0., a_max=0.999), dtype=torch.float32)

    def model(self, x: Tensor, cond: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError
    
    @staticmethod
    def extract(a: Tensor, t: Tensor, x_shape: Shape) -> Tensor:
        '''Gather per-batch timestep coefficients and reshape.'''
        out = a.gather(-1, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Optional[Tensor]=None) -> Tensor:
        '''Diffusion forward process: add noise to x_start at timestep t.'''
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        '''Use model output (predicted noise) to predict x_0.'''
        return (
            self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start: Tensor, x_t: Tensor, t: Tensor
                    ) -> Tuple[Tensor, Tensor, Tensor]:
        '''Compute mean and variance of q(x_{t-1} | x_t, x_0) for posterior sampling.'''
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x: Tensor, cond: Tensor, t: Tensor
                        ) -> Tuple[Tensor, Tensor, Tensor]:
        '''Use model output to compute mean and variance of p(x_{t-1} | x_t).'''
        x_recon = self.predict_start_from_noise(x_t=x, t=t, noise=self.model(x, cond, t))
        if self.clip_denoised:
            x_recon = x_recon.clamp(-1.0, 1.0)
        return self.q_posterior(x_recon, x, t)

    def p_sample(self, x: Tensor, cond: Tensor, t: Tensor) -> Tensor:
        model_mean, _, model_log_variance = self.p_mean_variance(x, cond, t)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().reshape(x.shape[0], *((1,) * (x.dim() - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    def p_sample_loop(self, shape: Tuple[int, ...], cond: Tensor) -> Tensor:
        x = torch.randn(shape, device=cond.device)
        for i in reversed(range(self.n_timesteps)):
            t = torch.full((shape[0],), i, device=cond.device, dtype=torch.long)
            x = self.p_sample(x, cond, t)
        return x

    def forward(self, z: Tensor) -> Tensor:
        '''Pass-through: z is the conditioning signal for compute_loss / decode_output.'''
        return z

    def compute_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        '''Compute simple noise prediction loss.'''
        x, cond = target, pred
        batch = x.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch,), device=x.device).long()
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise)
        x_recon = self.model(x_noisy, cond, t)
        return F.mse_loss(x_recon, noise)

    def decode_output(self, tensor: Tensor, **kwargs) -> Tensor:
        '''Reverse chain decode using Janner-style posterior sampling.'''
        cond = tensor
        shape = (cond.shape[0], *self.output_shape)
        x = self.p_sample_loop(shape, cond)
        return self._post_process(x)

    def _post_process(self, x: Tensor) -> Tensor:
        return x.float()


class VectorDiffusionDecoder(GaussianDiffusion):
    '''Diffusion decoder for vector real-valued states using an MLP denoiser.'''

    def __init__(self, output_shape: Shape, d_model: int, T: int=256, hidden: int=128) -> None:
        super().__init__(output_shape, d_model, T)
        self.clip_denoised = False

        data_dim = int(np.prod(output_shape, dtype=np.int64))
        self.time_emb = nn.Sequential(
            _SinusoidalPosEmb(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden + d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, data_dim),
        )

    def model(self, x: Tensor, cond: Tensor, t: Tensor) -> Tensor:
        flat = x.reshape(x.shape[0], -1)
        y = self.net(torch.cat([flat, self.time_emb(t), cond], dim=-1))
        return y.reshape(x.shape[0], *self.output_shape)


class ImageDiffusionDecoder(GaussianDiffusion):
    '''Diffusion decoder for image states using a UNet denoiser.'''

    def __init__(self, output_shape: Shape, d_model: int, T: int=256, base_ch: int=32) -> None:
        super().__init__(output_shape, d_model, T)
        self.clip_denoised = True
        
        # time and condition are summed into a single conditioning vector
        t_dim = base_ch * 4
        self.time_emb = nn.Sequential(
            _SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim), 
            nn.GELU()
        )
        self.cond_proj = nn.Linear(d_model, t_dim)

        # encoder
        c, _, _ = output_shape
        self.in_conv = nn.Conv2d(c, base_ch, 3, padding=1)
        self.enc0 = _ResidualBlock(base_ch, base_ch, t_dim)
        self.down0 = nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1)
        self.enc1 = _ResidualBlock(base_ch, base_ch * 2, t_dim)
        self.down1 = nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1)

        # bottleneck
        self.mid = _ResidualBlock(base_ch * 2, base_ch * 2, t_dim)

        # decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = _ResidualBlock(base_ch * 2 + base_ch * 2, base_ch, t_dim)
        self.up0 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec0 = _ResidualBlock(base_ch + base_ch, base_ch, t_dim)
        self.out = nn.Conv2d(base_ch, c, 1)
        
    def compute_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        return super().compute_loss(pred, target.float() * 2.0 - 1.0)

    def model(self, x: Tensor, cond: Tensor, t: Tensor) -> Tensor:
        cv = self.time_emb(t) + self.cond_proj(cond)          # (batch, t_dim)
        x = self.in_conv(x)                                   # (B, ch, H, W)
        s0 = self.enc0(x, cv)                                 # (B, ch, H, W)
        s1 = self.enc1(self.down0(s0), cv)                    # (B, ch*2, H/2, W/2)
        x = self.mid(self.down1(s1), cv)                      # (B, ch*2, H/4, W/4)
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1), cv)  # (B, ch, H/2, W/2)
        x = self.dec0(torch.cat([self.up0(x), s0], dim=1), cv)  # (B, ch, H, W)
        return self.out(x)                                    # (B, c, H, W)

    def _post_process(self, x: Tensor) -> Tensor:
        return (x * 0.5 + 0.5).clamp(0.0, 1.0).float()

