"""Typed run configuration, loadable from YAML and overridable from the CLI."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    train_file: str = "data/train/darcy_train.h5"
    test_file: str = "data/test/darcy_test.h5"
    val_fraction: float = 0.20      # of the training file -> 80/20 train/val
    split_seed: int = 1234          # seed for the train/val split (not the model seed)
    n_train_max: int | None = None  # cap sample count, for quick smoke runs
    batch_size: int = 16
    num_workers: int = 0            # >0 on Windows costs more than it saves here
    pin_memory: bool = True
    normalize_a: bool = True        # z-score the coefficient field
    normalize_u: bool = True        # z-score the solution field


@dataclass
class ModelConfig:
    # "fcn" -> coordinate MLP (classic PINN);  "fno" -> 2D Fourier neural operator
    name: str = "fno"

    # --- coordinate MLP (fcn) ---
    hidden_width: int = 128
    hidden_depth: int = 5           # number of hidden layers
    activation: str = "tanh"        # tanh | gelu | silu | sin
    fourier_features: int = 0       # 0 disables random Fourier feature encoding
    fourier_scale: float = 5.0
    points_per_sample: int = 4096   # collocation/data points drawn per field per step

    # --- Fourier neural operator (fno) ---
    modes: int = 12
    width: int = 32
    n_layers: int = 4
    lifting_width: int = 128
    include_grid: bool = True       # append (x, y) channels to the input
    # Pad the field before the spectral stack, crop after. The FFT assumes a
    # periodic domain; Darcy with u = 0 walls is not periodic, so without this
    # the wrap-around discontinuity rings back into the solution. 0 disables.
    domain_padding: int = 0


@dataclass
class PhysicsConfig:
    enabled: bool = False           # False -> the "without PDE" run
    beta: float = 1.0               # forcing f(x) = beta (PDEBench beta1.0 file)
    lambda_pde: float = 0.1         # weight on the PDE residual term
    # fcn only: weight on u = 0 at the Dirichlet ghost ring (the fno stencil
    # imposes the same condition through zero-padding, so it ignores this)
    lambda_bc: float = 1.0
    warmup_epochs: int = 5          # ramp the PDE weight in linearly over N epochs
    # fcn only: "mixed" first-order system (u, qx, qy) avoids second derivatives
    fcn_form: str = "mixed"         # mixed | second_order
    # finite-difference stencil for -div(a grad u) on the grid:
    # conservative_arith (arithmetic faces, matches PDEBench) | conservative
    # (harmonic faces) | expanded
    fd_scheme: str = "conservative_arith"


@dataclass
class OptimConfig:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"       # cosine | step | plateau | none
    step_size: int = 30             # step scheduler
    gamma: float = 0.5              # step scheduler
    min_lr: float = 1e-6
    grad_clip: float = 1.0          # 0 disables
    amp: bool = False               # keep off: PDE residuals need fp32 derivatives
    early_stop_patience: int = 0    # 0 disables early stopping


@dataclass
class RunConfig:
    name: str = "fno_nopde"
    seed: int = 0
    device: str = "auto"
    out_root: str = "."             # checkpoints/, logs/, results/ live under here
    log_every: int = 20             # batches between progress lines
    save_every: int = 0             # 0 -> only best + last

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    # ---- derived paths -------------------------------------------------
    @property
    def root(self) -> Path:
        p = Path(self.out_root)
        return p if p.is_absolute() else (ROOT / p).resolve()

    @property
    def ckpt_dir(self) -> Path:
        return self.root / "checkpoints" / self.name

    @property
    def log_dir(self) -> Path:
        return self.root / "logs" / self.name

    @property
    def result_dir(self) -> Path:
        return self.root / "results" / self.name

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (ROOT / p).resolve()


# ---------------------------------------------------------------------------
# (de)serialisation
# ---------------------------------------------------------------------------

def _from_dict(cls, data: dict) -> Any:
    """Recursively build a dataclass from a plain dict, rejecting unknown keys."""
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"unknown config key '{key}' for {cls.__name__}; "
                           f"expected one of {sorted(known)}")
        # dataclass-typed fields arrive as nested dicts
        target = {"data": DataConfig, "model": ModelConfig,
                  "physics": PhysicsConfig, "optim": OptimConfig}.get(key)
        kwargs[key] = (_from_dict(target, value)
                       if target is not None and isinstance(value, dict) else value)
    return cls(**kwargs)


def config_from_dict(data: dict) -> RunConfig:
    """Rebuild a RunConfig from the dict stored inside a checkpoint."""
    return _from_dict(RunConfig, data)


def load_config(path: str | Path) -> RunConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _from_dict(RunConfig, raw)


def apply_overrides(cfg: RunConfig, overrides: list[str]) -> RunConfig:
    """Apply `dotted.key=value` CLI overrides, e.g. `optim.epochs=5`."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override '{item}' is not of the form key=value")
        key, raw = item.split("=", 1)
        target: Any = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise KeyError(f"no config section '{part}' in '{key}'")
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise KeyError(f"no config field '{leaf}' in '{key}'")
        current = getattr(target, leaf)
        setattr(target, leaf, _coerce(raw, current))
    return cfg


def _coerce(raw: str, current: Any) -> Any:
    if raw.lower() in ("none", "null"):
        return None
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes", "y", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if current is None:                       # optional field: guess the type
        for cast in (int, float):
            try:
                return cast(raw)
            except ValueError:
                pass
        return raw
    return raw


def config_to_dict(cfg: Any) -> dict:
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return {f.name: config_to_dict(getattr(cfg, f.name)) for f in fields(cfg)}
    if isinstance(cfg, Path):
        return str(cfg)
    return cfg


def save_config(cfg: RunConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False)
