"""2D Fourier Neural Operator (Li et al., 2021), grid-to-grid.

    a(x)  [B, 1, H, W]   ->   u(x)  [B, 1, H, W]

Each layer applies a spectral convolution -- FFT, a learned complex multiplier
on the lowest `modes` frequencies, inverse FFT -- in parallel with a pointwise
1x1 convolution.  Truncating to low modes is what makes the operator resolution
agnostic: the learned weights live in frequency space, not on grid nodes, so the
same checkpoint can be evaluated on a finer or coarser mesh.

Inputs and outputs are physical; normalisation buffers live inside the module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Complex multiplication on the lowest `modes1` x `modes2` Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        # two blocks: positive and negative low frequencies along the first axis
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2,
                               dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2,
                               dtype=torch.cfloat))

    @staticmethod
    def _compl_mul2d(inp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # [B, Ci, M1, M2] x [Ci, Co, M1, M2] -> [B, Co, M1, M2]
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, hgt, wid = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # guard against grids smaller than the mode budget (e.g. 32x32 eval)
        m1 = min(self.modes1, hgt // 2)
        m2 = min(self.modes2, wid // 2 + 1)

        out_ft = torch.zeros(b, self.out_channels, hgt, wid // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m1, :m2] = self._compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])

        return torch.fft.irfft2(out_ft, s=(hgt, wid), norm="ortho")


class FNO2d(nn.Module):
    def __init__(self, modes: int = 12, width: int = 32, n_layers: int = 4,
                 lifting_width: int = 128, include_grid: bool = True,
                 domain_padding: int = 0, stats: dict | None = None) -> None:
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.include_grid = include_grid
        self.domain_padding = domain_padding

        stats = stats or {}
        self.register_buffer("a_mean", torch.tensor(float(stats.get("a_mean", 0.0))))
        self.register_buffer("a_std", torch.tensor(float(stats.get("a_std", 1.0))))
        self.register_buffer("u_mean", torch.tensor(float(stats.get("u_mean", 0.0))))
        self.register_buffer("u_std", torch.tensor(float(stats.get("u_std", 1.0))))

        in_ch = 1 + (2 if include_grid else 0)
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.pointwise = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.proj1 = nn.Conv2d(width, lifting_width, 1)
        self.proj2 = nn.Conv2d(lifting_width, 1, 1)

    @staticmethod
    def _grid(shape, device, dtype) -> torch.Tensor:
        b, _, hgt, wid = shape
        xs = torch.linspace(0, 1, wid, device=device, dtype=dtype)
        ys = torch.linspace(0, 1, hgt, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0)
        return grid.expand(b, -1, -1, -1)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """[B, 1, H, W] physical coefficient field -> [B, 1, H, W] physical u."""
        x = (a - self.a_mean) / self.a_std
        if self.include_grid:
            x = torch.cat([x, self._grid(a.shape, a.device, a.dtype)], dim=1)

        x = self.lift(x)

        # The FFT treats the domain as periodic, but Darcy on the unit square with
        # u = 0 walls is not: the left and right edges are unrelated. Padding the
        # field before the spectral stack and cropping after gives the transform a
        # buffer zone to absorb that discontinuity, instead of ringing back into
        # the solution. Standard practice for non-periodic FNO problems.
        p = self.domain_padding
        if p:
            x = F.pad(x, (0, p, 0, p))

        for i, (spec, pw) in enumerate(zip(self.spectral, self.pointwise)):
            x = spec(x) + pw(x)
            if i < self.n_layers - 1:       # no activation after the last block
                x = F.gelu(x)

        if p:
            x = x[..., :-p, :-p]

        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x * self.u_std + self.u_mean

    def predict_field(self, a_field: torch.Tensor, xs=None, ys=None) -> torch.Tensor:
        """Uniform interface with `CoordinateMLP.predict_field`."""
        return self.forward(a_field)
