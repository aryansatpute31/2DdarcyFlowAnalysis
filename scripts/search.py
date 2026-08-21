"""Hyperparameter search, ranked on validation only.

The test split is never read here. Configurations are scored on the 1,900-sample
validation fold at a short budget, ranked, and the winner is then trained to full
length by `train.py`. That ordering is what keeps the final test number honest --
tuning against the test set would make it a training metric.

    python scripts/search.py --space fno        # architecture sweep
    python scripts/search.py --space fcn
    python scripts/search.py --space lambda_fno # physics weight, given an arch
    python scripts/search.py --space fno --epochs 25 --list

Results append to `search/<space>/leaderboard.csv`, so an interrupted sweep can be
resumed by re-running (already-finished trials are skipped).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


# ---------------------------------------------------------------------------
# search spaces
# ---------------------------------------------------------------------------
# Each entry is (label, [config overrides]). Labels must be filesystem-safe.

def _fno_space() -> list[tuple[str, list[str]]]:
    """Capacity x spectral resolution x the non-periodic padding fix."""
    trials = []
    # baseline first, so the leaderboard always has the incumbent to beat
    trials.append(("base_m12_w32_p0", ["model.modes=12", "model.width=32",
                                       "model.domain_padding=0"]))
    for modes, width in [(12, 32), (16, 32), (16, 64), (20, 64), (24, 64)]:
        for pad in (8,):
            trials.append((f"m{modes}_w{width}_p{pad}",
                           [f"model.modes={modes}", f"model.width={width}",
                            f"model.domain_padding={pad}"]))
    # depth and learning rate, on a good mid config
    trials.append(("m16_w64_p8_L5", ["model.modes=16", "model.width=64",
                                     "model.domain_padding=8", "model.n_layers=5"]))
    trials.append(("m16_w64_p8_lr2e3", ["model.modes=16", "model.width=64",
                                        "model.domain_padding=8", "optim.lr=0.002"]))
    trials.append(("m16_w64_p8_lr5e4", ["model.modes=16", "model.width=64",
                                        "model.domain_padding=8", "optim.lr=0.0005"]))

    # Round 2, added after round 1 showed depth (L5) and a higher learning rate
    # (2e-3) each beat every capacity increase on their own -- and were never
    # tested together. Round 1 also found the mode ceiling: 24 was worse than 20.
    trials.append(("m16_w64_p8_L5_lr2e3", ["model.modes=16", "model.width=64",
                                           "model.domain_padding=8",
                                           "model.n_layers=5", "optim.lr=0.002"]))
    trials.append(("m20_w64_p8_L5_lr2e3", ["model.modes=20", "model.width=64",
                                           "model.domain_padding=8",
                                           "model.n_layers=5", "optim.lr=0.002"]))
    trials.append(("m16_w64_p8_lr3e3", ["model.modes=16", "model.width=64",
                                        "model.domain_padding=8", "optim.lr=0.003"]))
    return trials


def _fcn_space() -> list[tuple[str, list[str]]]:
    """Capacity, activation, and Fourier encoding of the coordinates.

    Worth stating up front: this architecture is a pointwise map and its error is
    dominated by that, not by capacity. These trials measure how much of the gap
    is reachable by tuning -- the honest answer is expected to be "not much".
    """
    trials = [
        ("base_d5_w128_tanh", ["model.hidden_depth=5", "model.hidden_width=128"]),
        ("d6_w256_tanh", ["model.hidden_depth=6", "model.hidden_width=256"]),
        ("d8_w256_tanh", ["model.hidden_depth=8", "model.hidden_width=256"]),
        ("d6_w512_tanh", ["model.hidden_depth=6", "model.hidden_width=512"]),
        ("d6_w256_gelu", ["model.hidden_depth=6", "model.hidden_width=256",
                          "model.activation=gelu"]),
        ("d6_w256_silu", ["model.hidden_depth=6", "model.hidden_width=256",
                          "model.activation=silu"]),
        ("d6_w256_ff32", ["model.hidden_depth=6", "model.hidden_width=256",
                          "model.fourier_features=32", "model.fourier_scale=5.0"]),
        ("d6_w256_ff64_s10", ["model.hidden_depth=6", "model.hidden_width=256",
                              "model.fourier_features=64", "model.fourier_scale=10.0"]),
        ("d6_w256_pts8192", ["model.hidden_depth=6", "model.hidden_width=256",
                             "model.points_per_sample=8192"]),
        ("d6_w256_lr3e3", ["model.hidden_depth=6", "model.hidden_width=256",
                           "optim.lr=0.003"]),
    ]
    return trials


def _lambda_space(arch: str) -> list[tuple[str, list[str]]]:
    """Re-tune the physics weight for whatever architecture won.

    The residual scale depends on the architecture, so a lambda tuned for the old
    one does not transfer.
    """
    lams = ["0.00001", "0.00003", "0.0001", "0.0003", "0.001", "0.003"]
    return [(f"lam{l}", [f"physics.lambda_pde={l}"]) for l in lams]


SPACES = {
    "fno": ("configs/fno_nopde.yaml", _fno_space),
    "fcn": ("configs/fcn_nopde.yaml", _fcn_space),
    "lambda_fno": ("configs/fno_pde.yaml", lambda: _lambda_space("fno")),
    "lambda_fcn": ("configs/fcn_pde.yaml", lambda: _lambda_space("fcn")),
}

FIELDS = ["trial", "val_rel_l2", "val_residual", "val_boundary", "params",
          "epochs", "seconds", "overrides"]


def read_done(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return {r["trial"] for r in csv.DictReader(fh)}


def best_epoch_row(history: Path) -> dict:
    with open(history, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return min(rows, key=lambda r: float(r["val_rel_l2"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, choices=sorted(SPACES))
    ap.add_argument("--epochs", type=int, default=20,
                    help="short budget per trial (default 20)")
    ap.add_argument("--n-train", type=int, default=0,
                    help="cap training samples; 0 uses all 7,600")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="overrides applied to every trial, e.g. model.modes=16")
    ap.add_argument("--list", action="store_true", help="print the space and exit")
    ap.add_argument("--rerun", action="store_true", help="ignore existing results")
    args = ap.parse_args()

    cfg_rel, space_fn = SPACES[args.space]
    trials = space_fn()

    if args.list:
        for name, ovr in trials:
            print(f"  {name:24s} {' '.join(ovr)}")
        print(f"\n{len(trials)} trials x {args.epochs} epochs")
        return 0

    out_root = ROOT / "search" / args.space
    out_root.mkdir(parents=True, exist_ok=True)
    board = out_root / "leaderboard.csv"
    done = set() if args.rerun else read_done(board)
    if not board.exists():
        with open(board, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, FIELDS).writeheader()

    print(f"=== search space '{args.space}' : {len(trials)} trials, "
          f"{args.epochs} epochs each ===")
    if done:
        print(f"    ({len(done)} already done, skipping; pass --rerun to redo)")

    for i, (name, ovr) in enumerate(trials, 1):
        if name in done:
            print(f"[{i}/{len(trials)}] {name:24s} -- skipped")
            continue

        run_name = f"{args.space}__{name}"
        cmd = [str(PYTHON), str(ROOT / "scripts" / "train.py"),
               "--config", str(ROOT / cfg_rel),
               f"name={run_name}",
               f"out_root={out_root}",
               f"optim.epochs={args.epochs}",
               "log_every=0"]
        if args.n_train:
            cmd.append(f"data.n_train_max={args.n_train}")
        cmd += ovr + args.extra

        print(f"[{i}/{len(trials)}] {name:24s} ", end="", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-4:]
            print("FAILED")
            for ln in tail:
                print("      " + ln)
            continue

        hist = out_root / "logs" / run_name / "history.csv"
        summ = out_root / "checkpoints" / run_name / "model_summary.json"
        try:
            row = best_epoch_row(hist)
            params = json.load(open(summ))["parameters_total"]
        except Exception as exc:
            print(f"FAILED to read results: {exc}")
            continue

        rec = {
            "trial": name,
            "val_rel_l2": f"{float(row['val_rel_l2']):.6f}",
            "val_residual": f"{float(row['val_pde_residual']):.4f}",
            "val_boundary": f"{float(row['val_boundary_rmse']):.4e}",
            "params": params,
            "epochs": args.epochs,
            "seconds": f"{dt:.0f}",
            "overrides": " ".join(ovr),
        }
        with open(board, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, FIELDS).writerow(rec)
        print(f"val rel-L2 {rec['val_rel_l2']}   ({params:,} params, {dt:.0f}s)")

    # ---- report ----------------------------------------------------------
    with open(board, newline="", encoding="utf-8") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: float(r["val_rel_l2"]))
    print(f"\n=== leaderboard: {args.space} (ranked on validation) ===")
    print(f"{'trial':26s}{'val rel-L2':>12}{'residual':>11}{'params':>12}")
    for r in rows:
        print(f"{r['trial']:26s}{float(r['val_rel_l2']):>12.6f}"
              f"{float(r['val_residual']):>11.2f}{int(r['params']):>12,}")
    if rows:
        b = rows[0]
        print(f"\nbest: {b['trial']}")
        print(f"  {b['overrides']}")
        base = next((r for r in rows if r["trial"].startswith("base")), None)
        if base and base is not b:
            gain = (float(base["val_rel_l2"]) - float(b["val_rel_l2"])) / float(base["val_rel_l2"])
            print(f"  {100 * gain:.1f} % better than the baseline trial "
                  f"({base['val_rel_l2']} -> {b['val_rel_l2']})")
    print(f"\n{board}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
