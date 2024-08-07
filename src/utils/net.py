import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.utils.common import weight_init

MLP_NORM = {"BN": nn.BatchNorm1d, "LN": nn.LayerNorm, "None": None}


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation=nn.ReLU,
        last_activation=False,
        norm="None",
        last_norm=False,
    ):
        super().__init__()
        norm = MLP_NORM[norm]
        self.output_dim = dims[-1]
        layers = []
        for k in range(len(dims) - 1):
            layers.append(nn.Linear(dims[k], dims[k + 1]))
            if norm is not None:
                layers.append(norm(dims[k + 1]))
            layers.append(activation())
        if not last_activation:
            layers.pop()
        if (norm is not None) and (not last_norm):
            layers.pop()
        self.mlp = nn.Sequential(*layers)
        self.apply(weight_init)
        # torch.compile(self.mlp, mode="max-autotune", fullgraph=True)

    def forward(self, obs):
        return self.mlp(obs.flatten(1))


def scale_obs(module, denom=255.0):
    class scaled_module(module):
        def forward(self, obs):
            return super().forward(obs / denom)

    return scaled_module


class CNN(nn.Module):
    def __init__(self, c, h, w, output_dim=None, scale=True, augment=False):
        super().__init__()
        self.scale = scale
        self.augment = augment
        print("scale is ", scale)
        self.input_shape = (c, h, w)
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            self.cnn_output_shape = self.cnn(torch.zeros(1, c, h, w)).shape[1:]
            self.cnn_output_dim = np.prod(self.cnn_output_shape)
            self.output_dim = self.cnn_output_dim
        self.encoder = nn.Sequential(self.cnn, nn.Flatten())
        if output_dim is not None:
            self.encoder = nn.Sequential(
                self.encoder,
                nn.Linear(self.cnn_output_dim, output_dim),
                nn.ReLU(inplace=True),
            )
            self.output_dim = output_dim
        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.output_dim, self.cnn_output_dim),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, self.cnn_output_shape),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, c, kernel_size=8, stride=4),
        )
        self.apply(weight_init)
        self.mse = nn.MSELoss()
        self.aug = RandomShiftsAug(4)

    def forward(self, obs):
        assert obs.shape[-3:] == self.input_shape
        x = obs / 255 - 0.5 if self.scale else obs
        return self.encoder(x)

    def loss(self, obs):
        assert obs.shape[-3:] == self.input_shape
        obs_aug = self.aug(obs) if self.augment else obs
        x_pred = self.decoder(self.decoder_mlp(self(obs_aug)))
        obs_pred = (x_pred + 0.5) * 255 if self.scale else x_pred
        return self.mse(obs_pred, obs_aug)


class Critic(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.mean = MLP(dims)

    def forward(self, obs):
        return self.mean(obs).flatten()


class DiscreteActor(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.mean = MLP(dims)

    def forward(self, obs, state=None):
        mean = self.mean(obs)
        return mean, state


class ContinuousActor(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.mean = MLP(dims)
        self.logstd = nn.Parameter(torch.zeros(1, dims[-1]))

    def forward(self, obs, state=None):
        mean = self.mean(obs)
        std = self.logstd.exp().expand_as(mean)
        return (mean, std), state


def get_encoder(obs_shape, mlp_hidden_dims, obs_rep_dim, mlp_norm, device, **kwargs):
    encoder = (
        MLP([obs_shape[0], *mlp_hidden_dims, obs_rep_dim], norm=mlp_norm)
        if len(obs_shape) == 1
        else CNN(*obs_shape, obs_rep_dim, scale=kwargs.get("scale", True))
    )
    return encoder.to(device)


def get_actor_critic(ac_input_dim, hidden_dims, act_dim, action_type, device):
    actor_dims = [ac_input_dim] + hidden_dims + [act_dim]
    actor = (
        DiscreteActor(actor_dims)
        if action_type == "discrete"
        else ContinuousActor(actor_dims)
    )
    critic = Critic([ac_input_dim] + hidden_dims + [1])
    return actor.to(device), critic.to(device)
