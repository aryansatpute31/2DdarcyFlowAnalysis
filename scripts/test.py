"""Evaluate a trained Darcy surrogate on the held-out 5 % test split.

    python scripts/test.py --run fno_pde
    python scripts/test.py --run all                    # every run + comparison table
    python scripts/test.py --run fcn_pde --save-predictions
    python scripts/test.py --checkpoint checkpoints/fno_pde/best.pt

The checkpoint is self-contained -- weights, normalisation statistics and the
full config all travel inside it -- so nothing here depends on the config files
still being present or unchanged since training.

Artefacts (under results/<run>/)
--------------------------------
test_metrics.json         summary statistics over the test set
per_sample_metrics.csv    one row per test sample, for digging into failures
figures/*.png             best / median / worst-case field comparisons
predictions.h5            predicted fields (only with --save-predictions)

With `--run all`, results/comparison.{csv,md,json} are written as well.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darcy.config import RunConfig, config_from_dict  # noqa: E402
from darcy.data import make_test_loader  # noqa: E402
from darcy.metrics import field_metrics, summarize  # noqa: E402
from darcy.models import build_model  # noqa: E402
from darcy.utils import get_logger, load_checkpoint, pick_device, save_json, set_seed  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:                                   # figures are a nicety
    HAVE_MPL = False

METRIC_KEYS = ["rel_l2", "rmse", "nrmse", "mae", "max_error",
               "boundary_rmse", "pde_residual_rmse", "pde_residual_rmse_truth"]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_run(ckpt_path: Path, device) -> tuple[torch.nn.Module, RunConfig, dict]:
    """Rebuild the model exactly as it was trained, straight from the checkpoint."""
    ck = load_checkpoint(ckpt_path, map_location=device)
    cfg = config_from_dict(ck["config"])
    stats = ck["normalizer"]
    model = build_model(cfg, stats).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, cfg, {"epoch": ck["epoch"], "best_val": ck["best_val"],
                        "stats": stats}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, grid, cfg: RunConfig, device,
             save_predictions: bool = False):
    xs = grid["x"].to(device)
    ys = grid["y"].to(device)
    h = float(grid["h"])
    beta = float(cfg.physics.beta)

    collected: dict[str, list[torch.Tensor]] = {}
    indices: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    total_time = 0.0
    n_seen = 0

    # warm up kernels / autotuning first, otherwise the first batch absorbs it
    # and the reported ms/sample is an order of magnitude too pessimistic
    warm = next(iter(loader))["a"][:2].to(device)
    for _ in range(3):
        model.predict_field(warm, xs, ys)
    if device.type == "cuda":
        torch.cuda.synchronize()

    for batch in loader:
        a = batch["a"].to(device, non_blocking=True)
        u = batch["u"].to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        u_pred = model.predict_field(a, xs, ys)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0
        n_seen += a.shape[0]

        m = field_metrics(u_pred, u, a, h, beta=beta,
                          scheme=cfg.physics.fd_scheme)
        for k, v in m.items():
            collected.setdefault(k, []).append(v.detach().cpu())
        indices.append(batch["index"].numpy())
        if save_predictions:
            preds.append(u_pred.detach().cpu().numpy().astype(np.float32))

    per_sample = {k: torch.cat(v).numpy() for k, v in collected.items()}
    summary = summarize(collected)
    summary["inference_ms_per_sample"] = 1000.0 * total_time / max(n_seen, 1)
    summary["n_test_samples"] = n_seen

    return summary, per_sample, np.concatenate(indices), (
        np.concatenate(preds) if save_predictions else None)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def plot_case(a, u_true, u_pred, title: str, out: Path) -> None:
    err = u_pred - u_true
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.1))
    panels = [
        (a, "coefficient  a(x)", "viridis", None),
        (u_true, "true  u(x)", "magma", None),
        (u_pred, "predicted  u(x)", "magma", None),
        (err, "error  (pred - true)", "coolwarm", "sym"),
    ]
    # share the colour scale between truth and prediction so they are comparable
    vmin, vmax = float(min(u_true.min(), u_pred.min())), float(max(u_true.max(), u_pred.max()))
    for ax, (data, label, cmap, mode) in zip(axes, panels):
        kw = {}
        if label.startswith(("true", "predicted")):
            kw = dict(vmin=vmin, vmax=vmax)
        elif mode == "sym":
            lim = float(np.abs(data).max()) or 1e-12
            kw = dict(vmin=-lim, vmax=lim)
        im = ax.imshow(data, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, **kw)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_error_distribution(rel_l2: np.ndarray, run: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rel_l2, bins=40, color="#4C78A8", edgecolor="white")
    ax.axvline(rel_l2.mean(), color="#E45756", lw=2,
               label=f"mean {rel_l2.mean():.4f}")
    ax.axvline(np.median(rel_l2), color="#F58518", lw=2, ls="--",
               label=f"median {np.median(rel_l2):.4f}")
    ax.set_xlabel("relative $L_2$ error")
    ax.set_ylabel("test samples")
    ax.set_title(f"{run} -- test error distribution")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def make_figures(run: str, loader, per_sample, model, grid, device, out_dir: Path,
                 n_cases: int = 3) -> None:
    rel = per_sample["rel_l2"]
    ds = loader.dataset
    order = np.argsort(rel)
    picks = {
        "best": order[:n_cases],
        "median": order[len(order) // 2: len(order) // 2 + n_cases],
        "worst": order[-n_cases:][::-1],
    }
    xs, ys = grid["x"].to(device), grid["y"].to(device)

    with torch.no_grad():
        for label, ids in picks.items():
            for rank, i in enumerate(ids):
                item = ds[int(i)]
                a = item["a"].unsqueeze(0).to(device)
                u = item["u"].unsqueeze(0).to(device)
                pred = model.predict_field(a, xs, ys)
                plot_case(
                    a[0, 0].cpu().numpy(), u[0, 0].cpu().numpy(),
                    pred[0, 0].cpu().numpy(),
                    f"{run} | {label} case #{rank} | test sample {int(i)} | "
                    f"rel-L2 = {rel[i]:.4f}",
                    out_dir / f"{label}_{rank}.png",
                )
    plot_error_distribution(rel, run, out_dir / "error_distribution.png")


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def print_summary(run: str, summary: dict, info: dict, n_params: int, log) -> None:
    log.info("")
    log.info("=" * 72)
    log.info(f"TEST RESULTS  --  {run}")
    log.info("=" * 72)
    log.info(f"  checkpoint epoch        : {info['epoch']}  "
             f"(best val rel-L2 {info['best_val']:.6f})")
    log.info(f"  parameters              : {n_params:,}")
    log.info(f"  test samples            : {summary['n_test_samples']}")
    log.info(f"  inference               : {summary['inference_ms_per_sample']:.3f} ms/sample")
    log.info("")
    log.info(f"  relative L2   mean      : {summary['rel_l2_mean']:.6f}")
    log.info(f"                median    : {summary['rel_l2_median']:.6f}")
    log.info(f"                p95       : {summary['rel_l2_p95']:.6f}")
    log.info(f"                max       : {summary['rel_l2_max']:.6f}")
    log.info(f"  RMSE          mean      : {summary['rmse_mean']:.6e}")
    log.info(f"  normalised RMSE         : {summary['nrmse_mean']:.6f}")
    log.info(f"  MAE           mean      : {summary['mae_mean']:.6e}")
    log.info(f"  max abs error mean      : {summary['max_error_mean']:.6e}")
    log.info(f"  boundary RMSE (u=0)     : {summary['boundary_rmse_mean']:.6e}")
    log.info(f"  PDE residual RMSE       : {summary['pde_residual_rmse_mean']:.6e}")
    log.info(f"    ... on ground truth   : {summary['pde_residual_rmse_truth_mean']:.6e}"
             "   <- discretisation noise floor")
    log.info("=" * 72)


def write_comparison(rows: list[dict], out_dir: Path, log) -> None:
    """Aggregate every evaluated run into one table."""
    cols = ["run", "model", "physics", "params", "test_rel_l2_mean",
            "test_rel_l2_median", "test_rmse", "test_nrmse",
            "boundary_rmse", "pde_residual_rmse", "best_val_rel_l2",
            "inference_ms"]
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in sorted(rows, key=lambda d: d["test_rel_l2_mean"]):
            w.writerow({k: r[k] for k in cols})

    lines = [
        "# Darcy flow surrogate comparison (held-out 5 % test split)", "",
        "| run | model | PDE | params | test rel-L2 | median | RMSE | nRMSE "
        "| boundary RMSE | PDE residual | val rel-L2 | ms/sample |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda d: d["test_rel_l2_mean"]):
        lines.append(
            f"| {r['run']} | {r['model']} | {'yes' if r['physics'] else 'no'} "
            f"| {r['params']:,} | **{r['test_rel_l2_mean']:.4f}** "
            f"| {r['test_rel_l2_median']:.4f} | {r['test_rmse']:.3e} "
            f"| {r['test_nrmse']:.4f} | {r['boundary_rmse']:.3e} "
            f"| {r['pde_residual_rmse']:.3e} | {r['best_val_rel_l2']:.4f} "
            f"| {r['inference_ms']:.3f} |")
    lines += ["", f"Ground-truth PDE residual (discretisation noise floor): "
                  f"{rows[0]['pde_residual_truth']:.3e}", ""]
    (out_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")

    save_json(rows, out_dir / "comparison.json")
    log.info("")
    log.info("\n".join(lines))
    log.info(f"[ok] comparison written to {out_dir / 'comparison.md'}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def evaluate_one(ckpt: Path, args, device, log) -> dict:
    model, cfg, info = load_run(ckpt, device)
    run = cfg.name
    n_params = sum(p.numel() for p in model.parameters())

    # the test file recorded in the checkpoint config, overridable from the CLI
    if args.test_file:
        cfg.data.test_file = str(args.test_file)
    loader, grid = make_test_loader(cfg, batch_size=args.batch_size)

    log.info(f"[..] {run}: {len(loader.dataset)} test samples from "
             f"{cfg.resolve(cfg.data.test_file).name}")

    summary, per_sample, indices, preds = evaluate(
        model, loader, grid, cfg, device, save_predictions=args.save_predictions)
    print_summary(run, summary, info, n_params, log)

    out_dir = cfg.result_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json({
        "run": run, "model": cfg.model.name,
        "physics_enabled": cfg.physics.enabled,
        "lambda_pde": cfg.physics.lambda_pde,
        "lambda_bc": cfg.physics.lambda_bc,
        "parameters": n_params,
        "checkpoint": str(ckpt),
        "checkpoint_epoch": info["epoch"],
        "best_val_rel_l2": info["best_val"],
        "test_file": str(cfg.resolve(cfg.data.test_file)),
        "grid_shape": list(grid["shape"]), "grid_spacing": grid["h"],
        "beta": cfg.physics.beta,
        "metrics": summary,
    }, out_dir / "test_metrics.json")

    # trace every row back to its row in the original PDEBench file
    with h5py.File(cfg.resolve(cfg.data.test_file), "r") as fh:
        raw_idx = (np.asarray(fh["source_index"][:]) if "source_index" in fh
                   else np.arange(len(indices)))

    with open(out_dir / "per_sample_metrics.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["test_index", "pdebench_index"] + METRIC_KEYS)
        for i in range(len(indices)):
            w.writerow([int(indices[i]), int(raw_idx[indices[i]])]
                       + [float(per_sample[k][i]) for k in METRIC_KEYS])

    if preds is not None:
        with h5py.File(out_dir / "predictions.h5", "w") as fh:
            fh.create_dataset("u_pred", data=preds, compression="lzf")
            fh.create_dataset("test_index", data=indices.astype("int64"))
            fh.attrs["run"] = run
        log.info(f"[ok] predictions -> {out_dir / 'predictions.h5'}")

    if HAVE_MPL and not args.no_figures:
        make_figures(run, loader, per_sample, model, grid, device,
                     out_dir / "figures", n_cases=args.n_figure_cases)
        log.info(f"[ok] figures -> {out_dir / 'figures'}")
    elif not HAVE_MPL:
        log.warning("matplotlib not installed; skipping figures "
                    "(pip install matplotlib)")

    log.info(f"[ok] metrics -> {out_dir / 'test_metrics.json'}")

    return {
        "run": run, "model": cfg.model.name, "physics": cfg.physics.enabled,
        "params": n_params,
        "test_rel_l2_mean": summary["rel_l2_mean"],
        "test_rel_l2_median": summary["rel_l2_median"],
        "test_rmse": summary["rmse_mean"],
        "test_nrmse": summary["nrmse_mean"],
        "boundary_rmse": summary["boundary_rmse_mean"],
        "pde_residual_rmse": summary["pde_residual_rmse_mean"],
        "pde_residual_truth": summary["pde_residual_rmse_truth_mean"],
        "best_val_rel_l2": info["best_val"],
        "inference_ms": summary["inference_ms_per_sample"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="run name under checkpoints/, or 'all'")
    g.add_argument("--checkpoint", type=Path, nargs="+",
                   help="path(s) to specific .pt files; several are compared")
    ap.add_argument("--which", default="best", choices=["best", "last"],
                    help="which checkpoint of the run to load (default: best)")
    ap.add_argument("--test-file", type=Path,
                    help="override the test HDF5 recorded in the checkpoint")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--n-figure-cases", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device("auto")
    log = get_logger("test", ROOT / "logs" / "test.log")
    log.info(f"device: {device}")

    if args.checkpoint:
        ckpts = list(args.checkpoint)
    elif args.run == "all":
        ckpts = sorted((ROOT / "checkpoints").glob(f"*/{args.which}.pt"))
        if not ckpts:
            log.error(f"no {args.which}.pt found under checkpoints/ -- train first")
            return 1
    else:
        ckpts = [ROOT / "checkpoints" / args.run / f"{args.which}.pt"]

    rows = []
    for ck in ckpts:
        if not ck.exists():
            log.error(f"checkpoint not found: {ck}")
            return 1
        rows.append(evaluate_one(ck, args, device, log))

    if len(rows) > 1:
        write_comparison(rows, ROOT / "results", log)

    return 0


if __name__ == "__main__":
    sys.exit(main())
