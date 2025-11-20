import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


def sinusoidal_time_embedding(t, dim: int):
    half_dim = dim // 2
    emb_scale = math.log(10000.0) / (half_dim - 1)
    freqs = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
    emb = t[:, None] * freqs[None, :]  # (B, half_dim)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.act = nn.GELU()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)

        return out + identity
    

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features, time_emb_dim=128):
        super().__init__()

        self.features = features
        self.time_emb_dim = time_emb_dim

        self.time_mlp_1 = nn.Linear(time_emb_dim, time_emb_dim)

        self.time_proj1 = nn.Linear(time_emb_dim, features)
        self.time_proj2 = nn.Linear(time_emb_dim, features * 2)

        self.down_conv1 = ResidualBlock(in_channels, features)
        self.down_conv2 = ResidualBlock(features, features * 2)

        self.bridge = ResidualBlock(features * 2, features * 4)

        self.up_conv2 = ResidualBlock(features * 6, features * 2)
        self.up_conv1 = ResidualBlock(features * 3, features)

        self.final_norm = nn.GroupNorm(8, features)
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=3, padding=1)

        self.act = nn.GELU()

    def _downsample(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=2)
    
    def _upsample(self, x, target_size):
        return F.interpolate(x, size=(target_size, target_size), mode='nearest')


    def forward(self, x, t):

        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp_1(t_emb)
        t_emb = self.act(t_emb)


        t_emb1 = self.time_proj1(t_emb)[:, :, None, None]   # (B, F, 1, 1)
        t_emb2 = self.time_proj2(t_emb)[:, :, None, None]   # (B, 2F, 1, 1)

        d1 = self.down_conv1(x)
        d1 = d1 + t_emb1.expand_as(d1)
        d2_in = self._downsample(d1)
        d2 = self.down_conv2(d2_in)
        d2 = d2 + t_emb2.expand_as(d2)

        b = self._downsample(d2)
        b = self.bridge(b)
 
        b_up = self._upsample(b, d2.shape[2])
        u2_in = torch.cat([b_up, d2], dim=1)
        u2 = self.up_conv2(u2_in)
        u2_up = self._upsample(u2, d1.shape[2])
        u1_in = torch.cat([u2_up, d1], dim=1)
        u1 = self.up_conv1(u1_in)

        out = self.final_norm(u1)
        out = self.act(out)
        out = self.final_conv(out)

        return out


class DiffusionModel:
    def __init__(self,
                 model: nn.Module,
                 num_steps: int,
                 beta_min: float,
                 beta_max: float,
                 device='cuda'):

        super().__init__()

        self.model = model
        self.num_steps = num_steps
        self.device = device

        self.time_schedule = self._time_schedule(num_steps).to(device)
        self.drift_schedule = self.subvp_drift_schedule(
            self.time_schedule, beta_min, beta_max).to(device)
        self.noise_schedule = self.subvp_noise_schedule(
            self.time_schedule, beta_min, beta_max).to(device)               

    def _time_schedule(self, num_steps: int) -> Tensor:
        t = torch.linspace(0, num_steps, num_steps + 1)
        return t / num_steps

    @staticmethod
    def subvp_drift_schedule(t: Tensor, beta_min: float = 0.1, beta_max: float = 20.0) -> Tensor:
        return torch.exp(-0.25 * (t ** 2) * (beta_max - beta_min) - 0.5 * t * beta_min)

    @staticmethod
    def subvp_noise_schedule(t: Tensor, beta_min: float = 0.1, beta_max: float = 20.0) -> Tensor:
        return 1.0 - torch.exp(-0.5 * (t ** 2) * (beta_max - beta_min) - t * beta_min)


    def forward(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """
        x: (B, C, H, W)
        t: (B,) integer steps
        """
        t = t.to(self.device)
        drift = self.drift_schedule[t].view(-1, 1, 1, 1)
        noise_scale = self.noise_schedule[t].view(-1, 1, 1, 1)

        noise = torch.randn_like(x)
        noisy_x = drift * x + noise_scale * noise
        return noisy_x, noise

    @torch.no_grad()
    def backward(self, x: Tensor) -> Tensor:

        x_t = x.clone().to(self.device)
        B = x_t.size(0)

        for t in reversed(range(self.num_steps)):

            dt = self.time_schedule[t] - self.time_schedule[t - 1]

            t_batch = torch.full((B,), t, dtype=torch.long, device=self.device)

            predicted_score = self.model(x_t, t_batch)

            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0.0

            drift_t = self.drift_schedule[t]
            noise_sched_t = self.noise_schedule[t]

            drift_t = drift_t.view(1, 1, 1, 1)
            noise_sched_t = noise_sched_t.view(1, 1, 1, 1)

            x_t = (
                x_t
                - dt * (drift_t * x_t + (noise_sched_t ** 2) * predicted_score)
                + math.sqrt(float(dt)) * noise
            )

        return x_t



if __name__ == "__main__":
    nn = UNet(in_channels=1, out_channels=1, features=32)

    model = DiffusionModel(model=nn, num_steps=5, beta_min=0.1, beta_max=20.0)

    x = torch.randn(1, 1, 32, 32)
    t = torch.tensor([10])  # example timestep
    out = model.forward(x, t)

    print(out)  # Expected output shape: (1, 1, 32, 32)