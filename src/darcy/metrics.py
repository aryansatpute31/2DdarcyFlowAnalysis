"""Losses and evaluation metrics.

The headline number for this benchmark is the **relative L2 error**, averaged
per sample:

    ||u_pred - u_true||_2 / ||u_true||_2

That is the metric the FNO and PDEBench papers report, so results here are
directly comparable to published numbers.  It is also scale free, which is why
it doubles as the data-fitting loss and lets us skip normalising the target.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .physics import darcy_residual_grid


class LpLoss(nn.Module):
    """Relative Lp loss, reduced per sample then averaged (or summed).

    `size_average=True` returns the mean over the batch; False returns the sum,
    which is what you want when accumulating over a loader with a ragged last
    batch.
    """

    def __init__(self, p: int = 2, size_average: bool = True, eps: float = 1e-8):
        super().__init__()
        self.p = p
        self.size_average = size_average
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b = pred.shape[0]
        diff = torch.norm(pred.reshape(b, -1) - target.reshape(b, -1),
                          p=self.p, dim=1)
        base = torch.norm(target.reshape(b, -1), p=self.p, dim=1)
        rel = diff / (base + self.eps)
        return rel.mean() if self.size_average else rel.sum()


def relative_l2_per_sample(pred: torch.Tensor, target: torch.Tensor,
                           eps: float = 1e-8) -> torch.Tensor:
    b = pred.shape[0]
    diff = torch.norm(pred.reshape(b, -1) - target.reshape(b, -1), p=2, dim=1)
    base = torch.norm(target.reshape(b, -1), p=2, dim=1)
    return diff / (base + eps)


def boundary_values(field: torch.Tensor) -> torch.Tensor:
    """Flatten the one-node-wide border ring of [B, 1, H, W] -> [B, P]."""
    top = field[..., 0, :]
    bottom = field[..., -1, :]
    left = field[..., 1:-1, 0]
    right = field[..., 1:-1, -1]
    return torch.cat([top.flatten(1), bottom.flatten(1),
                      left.flatten(1), right.flatten(1)], dim=1)


@torch.no_grad()
def field_metrics(pred: torch.Tensor, target: torch.Tensor, a: torch.Tensor,
                  h: float, beta: float = 1.0,
                  scheme: str = "conservative_arith") -> dict[str, torch.Tensor]:
    """Per-sample metrics for a batch of predicted fields.

    Every entry is a [B] tensor so the caller can concatenate across batches and
    report both the mean and the spread.

    `pde_residual_rmse` is computed with the same finite-difference operator for
    every architecture, so it is a common yardstick: it says how well a
    prediction satisfies the PDE regardless of how (or whether) that model was
    trained with a physics term.
    """
    b = pred.shape[0]
    err = pred - target

    rel_l2 = relative_l2_per_sample(pred, target)
    rmse = err.reshape(b, -1).pow(2).mean(dim=1).sqrt()
    mae = err.reshape(b, -1).abs().mean(dim=1)
    max_err = err.reshape(b, -1).abs().amax(dim=1)

    # normalised RMSE, relative to the dynamic range of each true field
    rng = (target.reshape(b, -1).amax(dim=1) - target.reshape(b, -1).amin(dim=1))
    nrmse = rmse / (rng + 1e-12)

    # error on the outermost ring.  Note this is |pred - true| there, NOT |pred|:
    # the grid is cell-centred, so the true solution is ~0.014 on that ring, not
    # zero.  u = 0 lives one spacing further out, on the ghost ring.
    bc_rmse = boundary_values(err).pow(2).mean(dim=1).sqrt()

    # the stencil zero-pads, so the residual is valid on the whole field
    res = darcy_residual_grid(pred, a, h, f=beta, scheme=scheme)
    res_rmse = res.reshape(b, -1).pow(2).mean(dim=1).sqrt()

    # same operator on the ground truth: the discretisation's own noise floor
    res_true = darcy_residual_grid(target, a, h, f=beta, scheme=scheme)
    res_true_rmse = res_true.reshape(b, -1).pow(2).mean(dim=1).sqrt()

    return {
        "rel_l2": rel_l2,
        "rmse": rmse,
        "nrmse": nrmse,
        "mae": mae,
        "max_error": max_err,
        "boundary_rmse": bc_rmse,
        "pde_residual_rmse": res_rmse,
        "pde_residual_rmse_truth": res_true_rmse,
    }


def summarize(collected: dict[str, list[torch.Tensor]]) -> dict[str, float]:
    """Concatenate per-batch metric tensors and reduce to mean/std/median/p95."""
    out: dict[str, float] = {}
    for key, chunks in collected.items():
        v = torch.cat(chunks).double()
        out[f"{key}_mean"] = v.mean().item()
        out[f"{key}_std"] = v.std(unbiased=False).item()
        out[f"{key}_median"] = v.median().item()
        out[f"{key}_p95"] = v.quantile(0.95).item()
        out[f"{key}_max"] = v.max().item()
    return out
