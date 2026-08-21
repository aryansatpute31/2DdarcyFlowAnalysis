"""Split the raw PDEBench Darcy file into train / test and sanity-check the physics.

Produces
--------
data/train/darcy_train.h5    95 % of the samples  (further split 80/20 at train time)
data/test/darcy_test.h5       5 % of the samples  (held out, never seen in training)
data/splits/split_manifest.json   the exact source indices, so the split is auditable

The split is a seeded permutation of the sample axis, so re-running with the same
`--seed` reproduces it exactly.

The script also runs a physics sanity check: it evaluates the finite-difference
Darcy operator on the *ground-truth* pairs (a, u).  The residual it reports is
the discretisation's own noise floor -- the best any model could hope to reach --
and it tells us which stencil best matches PDEBench's own solver.

Usage
-----
    python scripts/prepare_data.py
    python scripts/prepare_data.py --test-fraction 0.05 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

A_KEYS = ("nu", "a", "coefficient", "permeability")
U_KEYS = ("tensor", "u", "solution", "pressure")


def first_present(fh: h5py.File, keys: tuple[str, ...]) -> str:
    for k in keys:
        if k in fh:
            return k
    raise KeyError(f"none of {keys} in file; available: {list(fh.keys())}")


def squeeze_channel(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr[:, 0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


def find_coords(fh: h5py.File, n_x: int, n_y: int) -> tuple[np.ndarray, np.ndarray]:
    """Recover the node coordinates, falling back to a unit-square linspace."""
    for kx, ky in (("x-coordinate", "y-coordinate"), ("x", "y")):
        if kx in fh and ky in fh:
            return (np.asarray(fh[kx][:], dtype=np.float32),
                    np.asarray(fh[ky][:], dtype=np.float32))
    print("[warn] no coordinate arrays in file; assuming a unit-square linspace")
    return (np.linspace(0, 1, n_x, dtype=np.float32),
            np.linspace(0, 1, n_y, dtype=np.float32))


# ---------------------------------------------------------------------------
# physics sanity check
# ---------------------------------------------------------------------------

FACE_RULES = {
    "conservative_arith": lambda p, q: 0.5 * (p + q),
    "conservative": lambda p, q: 2.0 * p * q / (p + q + 1e-12),
}


def _residual_field(a: np.ndarray, u: np.ndarray, h: float, f: float,
                    face, u_pad: str) -> np.ndarray:
    """-div(a grad u) - f over the whole field, for a given ghost rule on u."""
    ap = np.pad(a, ((0, 0), (1, 1), (1, 1)), mode="edge")
    if u_pad == "dirichlet":
        up = np.pad(u, ((0, 0), (1, 1), (1, 1)), mode="constant")
    else:
        up = np.pad(u, ((0, 0), (1, 1), (1, 1)), mode="edge")

    ac, uc = ap[:, 1:-1, 1:-1], up[:, 1:-1, 1:-1]
    nb_a = [ap[:, 1:-1, 2:], ap[:, 1:-1, :-2], ap[:, 2:, 1:-1], ap[:, :-2, 1:-1]]
    nb_u = [up[:, 1:-1, 2:], up[:, 1:-1, :-2], up[:, 2:, 1:-1], up[:, :-2, 1:-1]]
    div = sum(face(ac, na) * (nu - uc) for na, nu in zip(nb_a, nb_u)) / (h * h)
    return -div - f


def residual_schemes(a: np.ndarray, u: np.ndarray, h: float,
                     f: float) -> dict[str, dict[str, float]]:
    """RMSE of -div(a grad u) - f for each stencil x ghost-rule combination.

    Two independent choices are being probed here:

    * the **face rule** -- how `a` is averaged onto cell faces;
    * the **ghost rule** for `u` outside the array.  PDEBench's grid is
      cell-centred, so the Dirichlet wall sits half a cell outside the stored
      data.  Zero-padding ("dirichlet") reproduces the generating solver;
      replicating the edge ("neumann") leaves a boundary residual four orders of
      magnitude too large.  This second choice matters far more than the first.
    """
    a = a.astype(np.float64)
    u = u.astype(np.float64)
    out: dict[str, dict[str, float]] = {}

    for fname, face in FACE_RULES.items():
        for u_pad in ("dirichlet", "neumann"):
            r = _residual_field(a, u, h, f, face, u_pad)
            ring = np.concatenate([r[:, 0, :].ravel(), r[:, -1, :].ravel(),
                                   r[:, 1:-1, 0].ravel(), r[:, 1:-1, -1].ravel()])
            out[f"{fname} / {u_pad}"] = {
                "rmse": float(np.sqrt((r ** 2).mean())),
                "median_abs": float(np.median(np.abs(r))),
                "boundary_rmse": float(np.sqrt((ring ** 2).mean())),
            }

    # the textbook expanded form, for reference (interior nodes only)
    ac = a[:, 1:-1, 1:-1]
    a_e, a_w = a[:, 1:-1, 2:], a[:, 1:-1, :-2]
    a_n, a_s = a[:, 2:, 1:-1], a[:, :-2, 1:-1]
    uc = u[:, 1:-1, 1:-1]
    u_e, u_w = u[:, 1:-1, 2:], u[:, 1:-1, :-2]
    u_n, u_s = u[:, 2:, 1:-1], u[:, :-2, 1:-1]
    lap = (u_e + u_w + u_n + u_s - 4 * uc) / (h * h)
    r = -(ac * lap + ((a_e - a_w) / (2 * h)) * ((u_e - u_w) / (2 * h))
          + ((a_n - a_s) / (2 * h)) * ((u_n - u_s) / (2 * h))) - f
    out["expanded / interior-only"] = {
        "rmse": float(np.sqrt((r ** 2).mean())),
        "median_abs": float(np.median(np.abs(r))),
        "boundary_rmse": float("nan"),
    }
    return out


def sanity_check(a: np.ndarray, u: np.ndarray, xs: np.ndarray, h: float,
                 beta: float) -> dict:
    print("\n" + "=" * 74)
    print("PHYSICS SANITY CHECK  (ground-truth pairs, so this is the noise floor)")
    print("=" * 74)
    print(f"  grid spacing h            : {h:.8f}   (1/{1/h:.2f})")
    cell_centred = abs(float(xs[0]) - h / 2) < 1e-6
    print(f"  x[0]                      : {float(xs[0]):.8f}   -> "
          f"{'CELL-CENTRED' if cell_centred else 'node-based'} grid")
    print(f"  a  range                  : [{a.min():.4e}, {a.max():.4e}]  "
          f"mean {a.mean():.4e}  std {a.std():.4e}")
    print(f"  u  range                  : [{u.min():.4e}, {u.max():.4e}]  "
          f"mean {u.mean():.4e}  std {u.std():.4e}")

    bnd = np.concatenate([u[:, 0, :].ravel(), u[:, -1, :].ravel(),
                          u[:, 1:-1, 0].ravel(), u[:, 1:-1, -1].ravel()])
    print(f"  |u| on stored ring (max)  : {np.abs(bnd).max():.4e}   "
          f"rmse {np.sqrt((bnd.astype(np.float64) ** 2).mean()):.4e}")
    if np.abs(bnd).max() > 1e-6:
        print("    ^ non-zero, as expected for a cell-centred grid: the Dirichlet")
        print("      wall sits outside the stored array, so the residual stencil")
        print("      must ZERO-PAD u rather than replicate the edge.")

    res = residual_schemes(a, u, h, beta)
    print(f"\n  residual of -div(a grad u) - {beta}, by stencil / ghost rule:")
    print(f"    {'scheme':<34}{'rmse':>12}{'median|r|':>12}{'boundary':>12}")
    ranked = sorted(res.items(), key=lambda kv: kv[1]["rmse"])
    for k, v in ranked:
        print(f"    {k:<34}{v['rmse']:>12.3e}{v['median_abs']:>12.3e}"
              f"{v['boundary_rmse']:>12.3e}")

    best = ranked[0][0]
    best_face = best.split(" / ")[0]
    print(f"\n  -> best: '{best}'")
    print(f"     use physics.fd_scheme={best_face}")
    print(f"     noise floor: rmse {ranked[0][1]['rmse']:.3e} "
          f"({100 * ranked[0][1]['rmse'] / beta:.2f} % of |f|), "
          f"median |r| {ranked[0][1]['median_abs']:.3e}")
    print("     no model can be expected to beat this floor.")
    print("=" * 74 + "\n")

    return {"h": h, "beta": beta, "cell_centred": cell_centred,
            "residuals": res, "best_scheme": best, "best_face_rule": best_face,
            "noise_floor_rmse": ranked[0][1]["rmse"],
            "stored_ring_abs_max": float(np.abs(bnd).max()),
            "a_min": float(a.min()), "a_max": float(a.max()),
            "a_unique": np.unique(a).tolist()[:8],
            "u_min": float(u.min()), "u_max": float(u.max())}


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def write_split(dest: Path, src: h5py.File, a_key: str, u_key: str,
                idx: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                beta: float, split_name: str, source_file: str,
                seed: int, chunk: int = 500) -> None:
    """Copy the selected samples into a fresh HDF5, in chunks to bound memory."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    idx = np.sort(np.asarray(idx))
    n = len(idx)
    hgt, wid = len(ys), len(xs)

    with h5py.File(dest, "w") as out:
        d_a = out.create_dataset("a", shape=(n, hgt, wid), dtype="float32",
                                 chunks=(min(32, n), hgt, wid),
                                 compression="lzf")
        d_u = out.create_dataset("u", shape=(n, hgt, wid), dtype="float32",
                                 chunks=(min(32, n), hgt, wid),
                                 compression="lzf")
        out.create_dataset("x", data=xs)
        out.create_dataset("y", data=ys)
        out.create_dataset("source_index", data=idx.astype("int64"))

        for start in range(0, n, chunk):
            sel = idx[start:start + chunk]
            d_a[start:start + len(sel)] = squeeze_channel(
                np.asarray(src[a_key][sel], dtype=np.float32))
            d_u[start:start + len(sel)] = squeeze_channel(
                np.asarray(src[u_key][sel], dtype=np.float32))
            print(f"\r  {split_name}: {min(start + chunk, n)}/{n}", end="", flush=True)
        print()

        out.attrs["beta"] = beta
        out.attrs["split"] = split_name
        out.attrs["n_samples"] = n
        out.attrs["source_file"] = source_file
        out.attrs["split_seed"] = seed
        out.attrs["pde"] = "-div(a grad u) = beta on (0,1)^2, u = 0 on boundary"
        out.attrs["citation"] = ("PDEBench Datasets, doi:10.18419/darus-2986 "
                                 "(Takamoto et al., NeurIPS 2022)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path,
                    default=ROOT / "data/raw/2D_DarcyFlow_beta1.0_Train.hdf5")
    ap.add_argument("--out-train", type=Path, default=ROOT / "data/train/darcy_train.h5")
    ap.add_argument("--out-test", type=Path, default=ROOT / "data/test/darcy_test.h5")
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "data/splits/split_manifest.json")
    ap.add_argument("--test-fraction", type=float, default=0.05,
                    help="fraction held out for testing (default 0.05 -> 95/5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--beta", type=float, default=1.0,
                    help="forcing term of the source file")
    ap.add_argument("--sanity-samples", type=int, default=64)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing split files")
    args = ap.parse_args()

    if not args.raw.exists():
        print(f"[fail] {args.raw} not found -- run scripts/download_data.py first")
        return 1
    for p in (args.out_train, args.out_test):
        if p.exists() and not args.force:
            print(f"[fail] {p} already exists; pass --force to overwrite")
            return 1

    with h5py.File(args.raw, "r") as fh:
        a_key = first_present(fh, A_KEYS)
        u_key = first_present(fh, U_KEYS)
        print(f"[..] source: {args.raw.name}")
        print(f"     keys  : a <- '{a_key}' {fh[a_key].shape}, "
              f"u <- '{u_key}' {fh[u_key].shape}")

        n_total = fh[a_key].shape[0]
        probe = squeeze_channel(np.asarray(fh[u_key][:1], dtype=np.float32))
        hgt, wid = probe.shape[-2], probe.shape[-1]
        xs, ys = find_coords(fh, wid, hgt)
        h = float(xs[1] - xs[0])

        # ---- sanity check on a random handful --------------------------
        rng = np.random.default_rng(args.seed)
        probe_idx = np.sort(rng.choice(n_total, min(args.sanity_samples, n_total),
                                       replace=False))
        a_probe = squeeze_channel(np.asarray(fh[a_key][probe_idx], dtype=np.float32))
        u_probe = squeeze_channel(np.asarray(fh[u_key][probe_idx], dtype=np.float32))
        diag = sanity_check(a_probe, u_probe, xs, h, args.beta)

        # ---- the 95/5 split --------------------------------------------
        perm = np.random.default_rng(args.seed).permutation(n_total)
        n_test = int(round(args.test_fraction * n_total))
        test_idx = np.sort(perm[:n_test])
        train_idx = np.sort(perm[n_test:])

        print(f"[..] splitting {n_total} samples -> "
              f"{len(train_idx)} train ({100 * len(train_idx) / n_total:.1f} %) / "
              f"{len(test_idx)} test ({100 * len(test_idx) / n_total:.1f} %)")
        assert not set(train_idx.tolist()) & set(test_idx.tolist()), "splits overlap"

        write_split(args.out_train, fh, a_key, u_key, train_idx, xs, ys,
                    args.beta, "train", args.raw.name, args.seed)
        write_split(args.out_test, fh, a_key, u_key, test_idx, xs, ys,
                    args.beta, "test", args.raw.name, args.seed)

    n_val = int(round(0.20 * len(train_idx)))
    manifest = {
        "source_file": str(args.raw),
        "source_doi": "doi:10.18419/darus-2986",
        "n_total": int(n_total),
        "grid": [int(hgt), int(wid)],
        "grid_spacing": h,
        "beta": args.beta,
        "split_seed": args.seed,
        "test_fraction": args.test_fraction,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "inner_split_note": (
            "train file is split 80/20 into train/val at run time by "
            "src/darcy/data.py using data.split_seed"),
        "n_train_inner": int(len(train_idx) - n_val),
        "n_val_inner": int(n_val),
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "diagnostics": diag,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[ok] {args.out_train}  ({len(train_idx)} samples)")
    print(f"[ok] {args.out_test}   ({len(test_idx)} samples)")
    print(f"[ok] {args.manifest}")
    print(f"\n  at train time the training file splits 80/20 -> "
          f"{len(train_idx) - n_val} train / {n_val} val")
    return 0


if __name__ == "__main__":
    sys.exit(main())
