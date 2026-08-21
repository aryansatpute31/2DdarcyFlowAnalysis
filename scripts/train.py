"""Train a Darcy surrogate: coordinate PINN or FNO, with or without the PDE term.

The four headline runs are just four config files:

    python scripts/train.py --config configs/fcn_nopde.yaml
    python scripts/train.py --config configs/fcn_pde.yaml
    python scripts/train.py --config configs/fno_nopde.yaml
    python scripts/train.py --config configs/fno_pde.yaml

Anything in the config can be overridden on the command line:

    python scripts/train.py --config configs/fno_pde.yaml optim.epochs=5 name=smoke

Data flow
---------
data/train/darcy_train.h5  (95 % of PDEBench)
        -> 80 % train fold   : gradient steps, and the source of the
                               normalisation statistics
        -> 20 % validation    : model selection only, never trained on

Artefacts
---------
checkpoints/<name>/best.pt         best validation relative-L2, with optimiser,
                                   normalisation buffers and config inside
checkpoints/<name>/last.pt         latest epoch, for resuming
checkpoints/<name>/config.yaml     the fully resolved config
checkpoints/<name>/model_summary.json
logs/<name>/train.log              human-readable log
logs/<name>/history.csv            one row per epoch
logs/<name>/metrics.json           final summary
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darcy.config import RunConfig, apply_overrides, load_config, save_config  # noqa: E402
from darcy.data import make_train_val_loaders  # noqa: E402
from darcy.metrics import LpLoss, field_metrics, summarize  # noqa: E402
from darcy.models import build_model  # noqa: E402
from darcy.physics import (  # noqa: E402
    coefficient_derivative_fields,
    darcy_residual_coords_mixed,
    darcy_residual_coords_second_order,
    darcy_residual_grid,
    gather_at,
    ghost_boundary_samples,
    node_coordinates,
    sample_interior_indices,
)
from darcy.utils import (  # noqa: E402
    AverageMeter,
    count_parameters,
    get_logger,
    load_checkpoint,
    pick_device,
    save_checkpoint,
    save_json,
    set_seed,
)

HISTORY_FIELDS = [
    "epoch", "lr", "train_loss", "train_data", "train_pde", "train_bc",
    "val_rel_l2", "val_rmse", "val_nrmse", "val_boundary_rmse",
    "val_pde_residual", "pde_weight", "epoch_seconds",
]


# ---------------------------------------------------------------------------
# per-architecture training step
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: RunConfig, model, grid, stats, device, logger):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.logger = logger
        self.stats = stats

        self.h = float(grid["h"])
        self.hgt, self.wid = grid["shape"]
        self.xs = grid["x"].to(device)
        self.ys = grid["y"].to(device)
        self.beta = float(cfg.physics.beta)

        self.is_fcn = cfg.model.name.lower() in ("fcn", "mlp", "pinn")
        self.lp = LpLoss(p=2)
        self.u_std = float(stats["u_std"]) or 1.0
        self.flux_scale = float(stats["flux_scale"]) or 1.0

    # -- loss pieces ------------------------------------------------------
    def pde_weight(self, epoch: int) -> float:
        """Linear warm-up of the physics weight over the first few epochs.

        Ramping in avoids the classic PINN failure where a large residual term
        dominates from step zero and the model collapses to a trivial field
        before it has learned anything from the data.
        """
        if not self.cfg.physics.enabled:
            return 0.0
        w = self.cfg.physics.warmup_epochs
        if w <= 0:
            return self.cfg.physics.lambda_pde
        return self.cfg.physics.lambda_pde * min(1.0, (epoch + 1) / w)

    # -- FNO --------------------------------------------------------------
    def _step_fno(self, batch, pde_w: float):
        a = batch["a"].to(self.device, non_blocking=True)
        u = batch["u"].to(self.device, non_blocking=True)

        u_pred = self.model(a)
        data_loss = self.lp(u_pred, u)

        pde_loss = torch.zeros((), device=self.device)
        bc_loss = torch.zeros((), device=self.device)
        if self.cfg.physics.enabled:
            # zero-padding inside the stencil already imposes u = 0 on the ghost
            # ring, so the Dirichlet condition needs no separate loss term here
            # and the residual is valid across the whole field, edges included
            res = darcy_residual_grid(u_pred, a, self.h, f=self.beta,
                                      scheme=self.cfg.physics.fd_scheme)
            pde_loss = (res / self.beta).pow(2).mean()

        total = data_loss + pde_w * pde_loss
        return total, data_loss.detach(), pde_loss.detach(), bc_loss.detach(), a.shape[0]

    # -- coordinate PINN --------------------------------------------------
    def _step_fcn(self, batch, pde_w: float):
        a = batch["a"].to(self.device, non_blocking=True)
        u = batch["u"].to(self.device, non_blocking=True)
        b = a.shape[0]
        p = self.cfg.model.points_per_sample
        mixed = self.cfg.physics.fcn_form == "mixed"

        idx = sample_interior_indices(b, self.hgt, self.wid, p, self.device)
        a_p = gather_at(a, idx)
        u_p = gather_at(u, idx)
        x_p, y_p = node_coordinates(idx, self.xs, self.ys, self.wid)

        need_grad = self.cfg.physics.enabled
        x_p = x_p.clone().requires_grad_(need_grad)
        y_p = y_p.clone().requires_grad_(need_grad)
        a_p = a_p.clone().requires_grad_(need_grad)

        out = self.model(x_p, y_p, a_p)
        u_pred = out[0] if isinstance(out, tuple) else out
        data_loss = self.lp(u_pred, u_p)

        pde_loss = torch.zeros((), device=self.device)
        bc_loss = torch.zeros((), device=self.device)

        if self.cfg.physics.enabled:
            d = coefficient_derivative_fields(a, self.h)
            a_x = gather_at(d["a_x"], idx)
            a_y = gather_at(d["a_y"], idx)

            if mixed:
                _, qx, qy = out
                r_x, r_y, r_m = darcy_residual_coords_mixed(
                    u_pred, qx, qy, x_p, y_p, a_p, a_x, a_y, f=self.beta)
                pde_loss = ((r_m / self.beta).pow(2).mean()
                            + (r_x / self.flux_scale).pow(2).mean()
                            + (r_y / self.flux_scale).pow(2).mean())
            else:
                a_xx = gather_at(d["a_xx"], idx)
                a_yy = gather_at(d["a_yy"], idx)
                r = darcy_residual_coords_second_order(
                    u_pred, x_p, y_p, a_p, a_x, a_y, a_xx, a_yy, f=self.beta)
                pde_loss = (r / self.beta).pow(2).mean()

            # Dirichlet u = 0 holds at the ghost ring one spacing outside the
            # cell-centred grid, not at the outermost stored node
            n_bc = max(p // 4, 64)
            gidx, x_b, y_b = ghost_boundary_samples(
                b, self.hgt, self.wid, n_bc, self.xs, self.ys, self.h, self.device)
            a_b = gather_at(a, gidx)
            out_b = self.model(x_b, y_b, a_b)
            u_b = out_b[0] if isinstance(out_b, tuple) else out_b
            bc_loss = (u_b / self.u_std).pow(2).mean()

        total = (data_loss + pde_w * pde_loss
                 + (self.cfg.physics.lambda_bc if self.cfg.physics.enabled else 0.0)
                 * bc_loss)
        return total, data_loss.detach(), pde_loss.detach(), bc_loss.detach(), b

    def step(self, batch, pde_w: float):
        return (self._step_fcn(batch, pde_w) if self.is_fcn
                else self._step_fno(batch, pde_w))

    # -- validation -------------------------------------------------------
    @torch.no_grad()
    def validate(self, loader) -> dict[str, float]:
        self.model.eval()
        collected: dict[str, list[torch.Tensor]] = {}
        for batch in loader:
            a = batch["a"].to(self.device, non_blocking=True)
            u = batch["u"].to(self.device, non_blocking=True)
            u_pred = self.model.predict_field(a, self.xs, self.ys)
            m = field_metrics(u_pred, u, a, self.h, beta=self.beta,
                              scheme=self.cfg.physics.fd_scheme)
            for k, v in m.items():
                collected.setdefault(k, []).append(v.detach().cpu())
        self.model.train()
        return summarize(collected)


# ---------------------------------------------------------------------------
# optimiser / scheduler
# ---------------------------------------------------------------------------

def build_optimizer(cfg: RunConfig, model):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr,
                            weight_decay=cfg.optim.weight_decay)
    s = cfg.optim.scheduler.lower()
    if s == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.optim.epochs, eta_min=cfg.optim.min_lr)
    elif s == "step":
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=cfg.optim.step_size, gamma=cfg.optim.gamma)
    elif s == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=cfg.optim.gamma, patience=5,
            min_lr=cfg.optim.min_lr)
    elif s in ("none", "constant"):
        sched = None
    else:
        raise ValueError(f"unknown scheduler '{cfg.optim.scheduler}'")
    return opt, sched


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--resume", action="store_true",
                    help="continue from checkpoints/<name>/last.pt")
    ap.add_argument("overrides", nargs="*",
                    help="dotted config overrides, e.g. optim.epochs=5")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)

    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"train.{cfg.name}", cfg.log_dir / "train.log")

    set_seed(cfg.seed)
    device = pick_device(cfg.device)

    log.info("=" * 70)
    log.info(f"run       : {cfg.name}")
    log.info(f"model     : {cfg.model.name}   physics: "
             f"{'ON' if cfg.physics.enabled else 'OFF'}")
    log.info(f"device    : {device}"
             + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    log.info(f"seed      : {cfg.seed}")
    log.info("=" * 70)

    if cfg.optim.amp and cfg.physics.enabled:
        log.warning("AMP is enabled alongside a PDE term; fp16 derivatives are "
                    "unreliable. Forcing amp=False.")
        cfg.optim.amp = False

    # -- data -------------------------------------------------------------
    t0 = time.time()
    train_loader, val_loader, stats, grid = make_train_val_loaders(cfg)
    n_tr, n_va = len(train_loader.dataset), len(val_loader.dataset)
    log.info(f"data      : {n_tr} train / {n_va} val "
             f"({100 * n_tr / (n_tr + n_va):.0f}/{100 * n_va / (n_tr + n_va):.0f} split, "
             f"seed {cfg.data.split_seed})  loaded in {time.time() - t0:.1f}s")
    log.info(f"grid      : {grid['shape']}  h = {grid['h']:.6f}  beta = {cfg.physics.beta}")
    log.info(f"stats     : a ~ N({stats['a_mean']:.4f}, {stats['a_std']:.4f}^2)   "
             f"u ~ N({stats['u_mean']:.4e}, {stats['u_std']:.4e}^2)   "
             f"flux_scale = {stats['flux_scale']:.4e}")

    # -- model ------------------------------------------------------------
    model = build_model(cfg, stats).to(device)
    total_p, train_p = count_parameters(model)
    log.info(f"params    : {total_p:,} total / {train_p:,} trainable")

    optimizer, scheduler = build_optimizer(cfg, model)
    trainer = Trainer(cfg, model, grid, stats, device, log)

    start_epoch, best_val = 0, float("inf")
    if args.resume and (cfg.ckpt_dir / "last.pt").exists():
        ck = load_checkpoint(cfg.ckpt_dir / "last.pt", map_location=device)
        model.load_state_dict(ck["model_state"])
        if ck.get("optimizer_state"):
            optimizer.load_state_dict(ck["optimizer_state"])
        if scheduler is not None and ck.get("scheduler_state"):
            scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch = ck["epoch"] + 1
        best_val = ck["best_val"]
        log.info(f"resumed from epoch {start_epoch} (best val rel-L2 {best_val:.6f})")

    save_config(cfg, cfg.ckpt_dir / "config.yaml")
    save_json({
        "run": cfg.name, "model": cfg.model.name,
        "physics_enabled": cfg.physics.enabled,
        "parameters_total": total_p, "parameters_trainable": train_p,
        "architecture": str(model),
        "normalization_stats": stats,
        "n_train": n_tr, "n_val": n_va,
        "grid_shape": list(grid["shape"]), "grid_spacing": grid["h"],
    }, cfg.ckpt_dir / "model_summary.json")

    hist_path = cfg.log_dir / "history.csv"
    if not args.resume or not hist_path.exists():
        with open(hist_path, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, HISTORY_FIELDS).writeheader()

    # -- loop -------------------------------------------------------------
    epochs_without_improvement = 0
    epoch = start_epoch - 1          # so the summary is sane if the loop is empty
    log.info(f"training for {cfg.optim.epochs} epochs "
             f"({len(train_loader)} batches/epoch)\n")

    for epoch in range(start_epoch, cfg.optim.epochs):
        model.train()
        pde_w = trainer.pde_weight(epoch)
        m_loss, m_data, m_pde, m_bc = (AverageMeter() for _ in range(4))
        t_epoch = time.time()

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            loss, l_data, l_pde, l_bc, n = trainer.step(batch, pde_w)

            if not torch.isfinite(loss):
                log.error(f"non-finite loss at epoch {epoch} batch {i}; "
                          "skipping this step")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            if cfg.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optimizer.step()

            m_loss.update(loss.item(), n)
            m_data.update(l_data.item(), n)
            m_pde.update(l_pde.item(), n)
            m_bc.update(l_bc.item(), n)

            if cfg.log_every and (i + 1) % cfg.log_every == 0:
                log.info(f"  e{epoch:03d} [{i + 1:4d}/{len(train_loader)}] "
                         f"loss {m_loss.avg:.5f}  data {m_data.avg:.5f}  "
                         f"pde {m_pde.avg:.4e}  bc {m_bc.avg:.4e}")

        val = trainer.validate(val_loader)
        lr_now = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val["rel_l2_mean"])
            else:
                scheduler.step()

        dt = time.time() - t_epoch
        log.info(
            f"epoch {epoch:03d}/{cfg.optim.epochs - 1} | lr {lr_now:.2e} | "
            f"train {m_loss.avg:.5f} (data {m_data.avg:.5f}, pde {m_pde.avg:.3e}, "
            f"bc {m_bc.avg:.3e}) | val rel-L2 {val['rel_l2_mean']:.5f} | "
            f"val res {val['pde_residual_rmse_mean']:.3e} | {dt:.1f}s"
        )

        with open(hist_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, HISTORY_FIELDS).writerow({
                "epoch": epoch, "lr": lr_now,
                "train_loss": m_loss.avg, "train_data": m_data.avg,
                "train_pde": m_pde.avg, "train_bc": m_bc.avg,
                "val_rel_l2": val["rel_l2_mean"], "val_rmse": val["rmse_mean"],
                "val_nrmse": val["nrmse_mean"],
                "val_boundary_rmse": val["boundary_rmse_mean"],
                "val_pde_residual": val["pde_residual_rmse_mean"],
                "pde_weight": pde_w, "epoch_seconds": dt,
            })

        ck_args = dict(model=model, optimizer=optimizer, scheduler=scheduler,
                       epoch=epoch, config=cfg, normalizer_state=stats,
                       extra={"val": val, "pde_weight": pde_w})
        save_checkpoint(cfg.ckpt_dir / "last.pt", best_val=best_val, **ck_args)

        if val["rel_l2_mean"] < best_val:
            best_val = val["rel_l2_mean"]
            save_checkpoint(cfg.ckpt_dir / "best.pt", best_val=best_val, **ck_args)
            log.info(f"  -> new best (val rel-L2 {best_val:.6f}), saved best.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if cfg.save_every and (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(cfg.ckpt_dir / f"epoch_{epoch:04d}.pt",
                            best_val=best_val, **ck_args)

        if (cfg.optim.early_stop_patience
                and epochs_without_improvement >= cfg.optim.early_stop_patience):
            log.info(f"early stop: no improvement for "
                     f"{epochs_without_improvement} epochs")
            break

    # -- wrap up ----------------------------------------------------------
    final_val = trainer.validate(val_loader)
    save_json({
        "run": cfg.name,
        "best_val_rel_l2": best_val,
        "final_val": final_val,
        "parameters_total": total_p,
        "epochs_run": epoch + 1,
        "physics_enabled": cfg.physics.enabled,
        "lambda_pde": cfg.physics.lambda_pde,
        "lambda_bc": cfg.physics.lambda_bc,
    }, cfg.log_dir / "metrics.json")

    log.info("")
    log.info(f"done. best val rel-L2 = {best_val:.6f}")
    log.info(f"  checkpoints -> {cfg.ckpt_dir}")
    log.info(f"  logs        -> {cfg.log_dir}")
    log.info(f"next: python scripts/test.py --run {cfg.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
