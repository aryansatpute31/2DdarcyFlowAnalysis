"""Dataset, splits and normalisation statistics.

Split policy
------------
`scripts/prepare_data.py` performs the outer **95 / 5 train / test** split once
and writes two files; the test file is then never touched during training.

`make_train_val_loaders` performs the inner **80 / 20 train / validation** split
on the training file at run time, seeded by `cfg.data.split_seed`.

Normalisation statistics are computed on the 80 % training fold *only* -- never
on validation or test -- so nothing leaks across the split boundary.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

A_KEYS = ("a", "nu", "coefficient", "permeability")
U_KEYS = ("u", "tensor", "solution", "pressure")


def _first_present(h5: h5py.File, keys) -> str:
    for k in keys:
        if k in h5:
            return k
    raise KeyError(f"none of {keys} found in file; available: {list(h5.keys())}")


def _squeeze_channel(arr: np.ndarray) -> np.ndarray:
    """PDEBench stores the solution as [N, 1, H, W]; collapse to [N, H, W]."""
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr[:, 0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


class DarcyDataset(Dataset):
    """Coefficient / pressure field pairs from a prepared HDF5 file.

    Yields `a` and `u` as [1, H, W] float32 tensors -- channel-first, which is
    what the FNO wants and what the finite-difference stencils in `physics.py`
    assume.
    """

    def __init__(self, path: str | Path, indices: np.ndarray | None = None,
                 in_memory: bool = True) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found -- run scripts/prepare_data.py first")

        with h5py.File(self.path, "r") as fh:
            a_key = _first_present(fh, A_KEYS)
            u_key = _first_present(fh, U_KEYS)
            n_total = fh[a_key].shape[0]
            self.x = np.asarray(fh["x"][:], dtype=np.float32)
            self.y = np.asarray(fh["y"][:], dtype=np.float32)
            self.beta = float(fh.attrs.get("beta", 1.0))

            self.indices = (np.arange(n_total) if indices is None
                            else np.asarray(indices))
            self.in_memory = in_memory
            if in_memory:
                order = np.sort(self.indices)
                self._a = _squeeze_channel(
                    np.asarray(fh[a_key][order], dtype=np.float32))
                self._u = _squeeze_channel(
                    np.asarray(fh[u_key][order], dtype=np.float32))
                # remap so __getitem__ can index positionally
                self._pos = {int(g): i for i, g in enumerate(order)}
            else:
                self._a = self._u = self._pos = None
                self._a_key, self._u_key = a_key, u_key
                self._fh = None

        self.h = float(self.x[1] - self.x[0])
        self.grid_shape = (len(self.y), len(self.x))

    def __len__(self) -> int:
        return len(self.indices)

    def _lazy_file(self) -> h5py.File:
        if self._fh is None:
            self._fh = h5py.File(self.path, "r")
        return self._fh

    def __getitem__(self, i: int) -> dict:
        gid = int(self.indices[i])
        if self.in_memory:
            p = self._pos[gid]
            a = self._a[p]
            u = self._u[p]
        else:
            fh = self._lazy_file()
            a = np.asarray(fh[self._a_key][gid], dtype=np.float32)
            u = _squeeze_channel(
                np.asarray(fh[self._u_key][gid], dtype=np.float32)[None])[0]
        return {
            "a": torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0),
            "u": torch.from_numpy(np.ascontiguousarray(u)).unsqueeze(0),
            "index": gid,
        }


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def compute_stats(dataset: DarcyDataset, max_samples: int = 2048,
                  seed: int = 0) -> dict:
    """Normalisation constants from a subsample of `dataset`.

    Returns a_mean/a_std/u_mean/u_std plus `flux_scale`, the typical magnitude
    of the Darcy flux q = -a grad u.  The coordinate MLP predicts fluxes in
    units of `flux_scale` so that its three outputs share a comparable scale and
    the optimiser is not fighting a badly conditioned problem.
    """
    n = len(dataset)
    rng = np.random.default_rng(seed)
    take = np.arange(n) if n <= max_samples else rng.choice(n, max_samples, False)

    a_sum = a_sq = u_sum = u_sq = 0.0
    count = 0
    flux_sq = 0.0
    flux_count = 0
    h = dataset.h

    for i in take:
        item = dataset[int(i)]
        a = item["a"].numpy()[0]
        u = item["u"].numpy()[0]
        a_sum += a.sum(dtype=np.float64)
        a_sq += (a.astype(np.float64) ** 2).sum()
        u_sum += u.sum(dtype=np.float64)
        u_sq += (u.astype(np.float64) ** 2).sum()
        count += a.size

        ux = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * h)
        uy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * h)
        ac = a[1:-1, 1:-1]
        flux_sq += ((ac * ux) ** 2 + (ac * uy) ** 2).astype(np.float64).sum()
        flux_count += 2 * ux.size

    a_mean = a_sum / count
    a_std = float(np.sqrt(max(a_sq / count - a_mean ** 2, 1e-12)))
    u_mean = u_sum / count
    u_std = float(np.sqrt(max(u_sq / count - u_mean ** 2, 1e-12)))
    flux_scale = float(np.sqrt(flux_sq / max(flux_count, 1)))

    return {
        "a_mean": float(a_mean), "a_std": a_std,
        "u_mean": float(u_mean), "u_std": u_std,
        "flux_scale": max(flux_scale, 1e-8),
        "n_samples_used": int(len(take)),
    }


def apply_stat_switches(stats: dict, normalize_a: bool, normalize_u: bool) -> dict:
    """Honour the `normalize_a` / `normalize_u` config switches.

    Disabling a switch collapses that transform to the identity while leaving the
    rest of the stats dict intact.
    """
    out = dict(stats)
    if not normalize_a:
        out["a_mean"], out["a_std"] = 0.0, 1.0
    if not normalize_u:
        out["u_mean"], out["u_std"] = 0.0, 1.0
    return out


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def split_train_val(n: int, val_fraction: float, seed: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 80/20-style index split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(val_fraction * n))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def make_train_val_loaders(cfg, in_memory: bool = True):
    """Build the training and validation loaders plus the normalisation stats."""
    path = cfg.resolve(cfg.data.train_file)

    with h5py.File(path, "r") as fh:
        n_total = fh[_first_present(fh, A_KEYS)].shape[0]
    if cfg.data.n_train_max is not None:
        n_total = min(n_total, int(cfg.data.n_train_max))

    tr_idx, va_idx = split_train_val(n_total, cfg.data.val_fraction,
                                     cfg.data.split_seed)

    train_ds = DarcyDataset(path, tr_idx, in_memory=in_memory)
    val_ds = DarcyDataset(path, va_idx, in_memory=in_memory)

    stats = compute_stats(train_ds, seed=cfg.data.split_seed)
    stats = apply_stat_switches(stats, cfg.data.normalize_a, cfg.data.normalize_u)

    common = dict(num_workers=cfg.data.num_workers,
                  pin_memory=cfg.data.pin_memory and torch.cuda.is_available(),
                  persistent_workers=cfg.data.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size,
                              shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size,
                            shuffle=False, drop_last=False, **common)

    grid = {
        "h": train_ds.h,
        "shape": train_ds.grid_shape,
        "x": torch.from_numpy(train_ds.x),
        "y": torch.from_numpy(train_ds.y),
        "beta": train_ds.beta,
    }
    return train_loader, val_loader, stats, grid


def make_test_loader(cfg, in_memory: bool = True, batch_size: int | None = None):
    path = cfg.resolve(cfg.data.test_file)
    ds = DarcyDataset(path, None, in_memory=in_memory)
    loader = DataLoader(ds, batch_size=batch_size or cfg.data.batch_size,
                        shuffle=False, drop_last=False,
                        num_workers=cfg.data.num_workers,
                        pin_memory=cfg.data.pin_memory and torch.cuda.is_available())
    grid = {
        "h": ds.h,
        "shape": ds.grid_shape,
        "x": torch.from_numpy(ds.x),
        "y": torch.from_numpy(ds.y),
        "beta": ds.beta,
    }
    return loader, grid
