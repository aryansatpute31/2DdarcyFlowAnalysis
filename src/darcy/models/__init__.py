"""Model registry.

Both architectures share one contract so the training and testing scripts can
stay architecture-agnostic:

* they consume **physical** (un-normalised) inputs,
* they normalise internally using buffers baked into the checkpoint,
* they return **physical** outputs.

Keeping normalisation inside the module means a checkpoint is self-contained:
loading it is enough to make correct predictions, with no separate stats file to
lose track of.
"""

from __future__ import annotations

import torch

from ..config import RunConfig
from .fcn import CoordinateMLP
from .fno import FNO2d

__all__ = ["CoordinateMLP", "FNO2d", "build_model"]


def build_model(cfg: RunConfig, stats: dict) -> torch.nn.Module:
    """Instantiate the architecture named in `cfg.model.name`.

    `stats` carries the training-set normalisation constants
    (`a_mean`, `a_std`, `u_mean`, `u_std`, `flux_scale`).
    """
    name = cfg.model.name.lower()
    m = cfg.model

    if name in ("fcn", "mlp", "pinn"):
        return CoordinateMLP(
            width=m.hidden_width,
            depth=m.hidden_depth,
            activation=m.activation,
            fourier_features=m.fourier_features,
            fourier_scale=m.fourier_scale,
            # the flux head only earns its keep when a mixed-form PDE term is
            # actually being optimised; otherwise it would be dead weight
            predict_flux=(cfg.physics.enabled and cfg.physics.fcn_form == "mixed"),
            stats=stats,
        )

    if name == "fno":
        return FNO2d(
            modes=m.modes,
            width=m.width,
            n_layers=m.n_layers,
            lifting_width=m.lifting_width,
            include_grid=m.include_grid,
            domain_padding=m.domain_padding,
            stats=stats,
        )

    raise ValueError(f"unknown model '{cfg.model.name}' (expected 'fcn' or 'fno')")
