"""Coordinate MLP -- the classic PINN architecture.

The network is a pointwise map

    N( x, y, a(x,y) )  ->  u(x,y)          [ and, in mixed mode, qx, qy ]

evaluated independently at every collocation point, exactly as in the original
PINN formulation.  Spatial derivatives come from autograd on the (x, y) inputs.

One property worth being explicit about: this is a *local* map.  Two different
permeability fields that happen to share the same value at a point receive the
same prediction there, so the network can only ever learn the conditional mean
E[ u | x, y, a(x,y) ].  Darcy flow is elliptic and therefore globally coupled,
so a pointwise model is structurally limited on this task and is expected to
trail the FNO.  That gap is the interesting result, not a bug -- it is the
reason neural *operators* exist.

Inputs and outputs are physical; normalisation happens inside via buffers, so a
checkpoint needs no companion stats file.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "sin": Sine,
}


class CoordinateMLP(nn.Module):
    """Pointwise MLP over (x, y, a).

    Parameters
    ----------
    predict_flux : bool
        When True the network emits (u, qx, qy) so the PDE residual can use the
        first-order mixed form and avoid second derivatives entirely.
    fourier_features : int
        If > 0, encode (x, y) with that many random Fourier features before the
        first layer.  Off by default to keep the baseline a textbook PINN.
    """

    def __init__(self, width: int = 128, depth: int = 5, activation: str = "tanh",
                 fourier_features: int = 0, fourier_scale: float = 5.0,
                 predict_flux: bool = True, stats: dict | None = None) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation '{activation}'; "
                             f"choose from {sorted(_ACTIVATIONS)}")

        self.predict_flux = predict_flux
        self.n_out = 3 if predict_flux else 1
        self.fourier_features = fourier_features

        stats = stats or {}
        self.register_buffer("a_mean", torch.tensor(float(stats.get("a_mean", 0.0))))
        self.register_buffer("a_std", torch.tensor(float(stats.get("a_std", 1.0))))
        self.register_buffer("u_mean", torch.tensor(float(stats.get("u_mean", 0.0))))
        self.register_buffer("u_std", torch.tensor(float(stats.get("u_std", 1.0))))
        self.register_buffer("flux_scale",
                             torch.tensor(float(stats.get("flux_scale", 1.0))))

        if fourier_features > 0:
            # fixed (non-trainable) random Fourier features on the coordinates
            b = torch.randn(2, fourier_features) * fourier_scale
            self.register_buffer("ff_b", b)
            in_dim = 2 * fourier_features + 1        # sin/cos bank + a
        else:
            self.ff_b = None
            in_dim = 3                               # x, y, a

        act = _ACTIVATIONS[activation]
        layers: list[nn.Module] = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, self.n_out)]
        self.net = nn.Sequential(*layers)

        self._init_weights(activation)

    def _init_weights(self, activation: str) -> None:
        gain = nn.init.calculate_gain("tanh") if activation == "tanh" else 1.0
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    # -- encoding ---------------------------------------------------------
    def _encode(self, x: torch.Tensor, y: torch.Tensor,
                a: torch.Tensor) -> torch.Tensor:
        """Physical (x, y, a) -> network input features.

        Kept inside the graph so autograd through `x`/`y`/`a` yields derivatives
        with respect to the *physical* variables and no manual chain rule is
        needed anywhere else.
        """
        a_n = (a - self.a_mean) / self.a_std
        if self.ff_b is not None:
            xy = torch.stack([x, y], dim=-1)               # [..., 2]
            proj = 2.0 * math.pi * (xy @ self.ff_b)        # [..., F]
            feats = [torch.sin(proj), torch.cos(proj), a_n.unsqueeze(-1)]
        else:
            # map the unit square to [-1, 1], which suits tanh far better
            feats = [(2.0 * x - 1.0).unsqueeze(-1),
                     (2.0 * y - 1.0).unsqueeze(-1),
                     a_n.unsqueeze(-1)]
        return torch.cat(feats, dim=-1)

    # -- forward ----------------------------------------------------------
    def forward(self, x: torch.Tensor, y: torch.Tensor, a: torch.Tensor):
        """Evaluate at arbitrary points.

        Parameters
        ----------
        x, y, a : broadcastable tensors of identical shape, e.g. [B, P].
                  Physical units; x, y in [0, 1].

        Returns
        -------
        u                       when predict_flux is False
        (u, qx, qy)             when predict_flux is True

        All outputs are physical:  u is de-normalised with the stored stats and
        the fluxes are rescaled by `flux_scale`.
        """
        out = self.net(self._encode(x, y, a))
        u = out[..., 0] * self.u_std + self.u_mean
        if not self.predict_flux:
            return u
        qx = out[..., 1] * self.flux_scale
        qy = out[..., 2] * self.flux_scale
        return u, qx, qy

    @torch.no_grad()
    def predict_field(self, a_field: torch.Tensor, xs: torch.Tensor,
                      ys: torch.Tensor, chunk: int = 1 << 20) -> torch.Tensor:
        """Dense evaluation on a full grid, for validation and testing.

        Parameters
        ----------
        a_field : [B, 1, H, W] physical coefficient field.
        xs, ys  : [W] and [H] physical node coordinates.

        Returns
        -------
        [B, 1, H, W] predicted pressure field.
        """
        b, _, hgt, wid = a_field.shape
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        xx = xx.reshape(1, -1).expand(b, -1)
        yy = yy.reshape(1, -1).expand(b, -1)
        aa = a_field.reshape(b, -1)

        outs = []
        for start in range(0, aa.shape[1], chunk):
            sl = slice(start, start + chunk)
            res = self.forward(xx[:, sl], yy[:, sl], aa[:, sl])
            outs.append(res[0] if self.predict_flux else res)
        return torch.cat(outs, dim=1).reshape(b, 1, hgt, wid)
