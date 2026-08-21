"""Verify the PDE machinery against cases with known answers.

Run this after changing anything in `src/darcy/physics.py`:

    python scripts/verify_physics.py

Four checks:

1. `total_grad` reproduces exact total derivatives.
   The coordinate PINN is N(x, y, a(x,y)), so du/dx = dN/dx + (dN/da)*a_x.
   We pick an N and an a(x,y) whose composition has a closed form, get the
   ground truth by autograd on that composition, and demand agreement.
   Dropping the chain-rule term fails this test loudly, which is the point.

2. Second total derivatives, same idea, including the (dN/da)*a_xx correction.

3. `darcy_residual_grid` against a manufactured solution, checking that the
   error falls at the second-order rate the stencil claims.

4. The mixed first-order system is consistent with the second-order form:
   feeding it the exact flux q = -a grad u must drive all three residuals to
   zero.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darcy.physics import (  # noqa: E402
    darcy_residual_coords_mixed,
    darcy_residual_grid,
    interior_mask,
    total_grad,
)

PI = math.pi
DTYPE = torch.float64          # float64 so discretisation error dominates, not noise
PASS, FAIL = "  PASS", "  FAIL"
_failures = 0


def report(name: str, ok: bool, detail: str) -> None:
    global _failures
    print(f"{PASS if ok else FAIL}  {name}\n         {detail}")
    if not ok:
        _failures += 1


# --- the analytic setup shared by checks 1, 2 and 4 ------------------------
# a(x, y) = 1 + x^2 + y^2         (smooth, so a_x and a_xx are exact)
# N(x, y, a) = sin(pi x) sin(pi y) / a
# => u(x, y) = sin(pi x) sin(pi y) / (1 + x^2 + y^2)

def a_of(x, y):
    return 1.0 + x ** 2 + y ** 2


def net(x, y, a):
    """Stands in for the network: an explicit function of all three inputs."""
    return torch.sin(PI * x) * torch.sin(PI * y) / a


def composite(x, y):
    """The same thing with a substituted inline -- a pure function of (x, y)."""
    return net(x, y, a_of(x, y))


def check_first_derivatives() -> None:
    torch.manual_seed(0)
    n = 512
    x0 = torch.rand(n, dtype=DTYPE)
    y0 = torch.rand(n, dtype=DTYPE)

    # ground truth: autograd straight through the composition
    xc = x0.clone().requires_grad_(True)
    yc = y0.clone().requires_grad_(True)
    uc = composite(xc, yc)
    gx_true, gy_true = torch.autograd.grad(uc, (xc, yc),
                                           torch.ones_like(uc), create_graph=True)

    # what the training code actually does: a is a separate leaf, a_x supplied
    x = x0.clone().requires_grad_(True)
    y = y0.clone().requires_grad_(True)
    a = a_of(x0, y0).clone().requires_grad_(True)
    u = net(x, y, a)
    gx, gy, _ = total_grad(u, x, y, a, a_x=2 * x0, a_y=2 * y0)

    e_x = (gx - gx_true).abs().max().item()
    e_y = (gy - gy_true).abs().max().item()
    report("total_grad matches exact du/dx, du/dy",
           max(e_x, e_y) < 1e-10,
           f"max |err| = {e_x:.3e} (x), {e_y:.3e} (y)")

    # and confirm the chain-rule term is doing real work
    naive = torch.autograd.grad(net(x, y, a), x, torch.ones_like(u),
                                retain_graph=True)[0]
    gap = (naive - gx_true).abs().max().item()
    report("chain-rule term is non-negligible (a naive dN/dx would be wrong)",
           gap > 1e-3, f"ignoring it would introduce max |err| = {gap:.3e}")


def check_second_derivatives() -> None:
    torch.manual_seed(0)
    n = 512
    x0 = torch.rand(n, dtype=DTYPE)
    y0 = torch.rand(n, dtype=DTYPE)

    xc = x0.clone().requires_grad_(True)
    yc = y0.clone().requires_grad_(True)
    uc = composite(xc, yc)
    gx_true = torch.autograd.grad(uc, xc, torch.ones_like(uc), create_graph=True)[0]
    gxx_true = torch.autograd.grad(gx_true, xc, torch.ones_like(gx_true),
                                   create_graph=True)[0]

    x = x0.clone().requires_grad_(True)
    y = y0.clone().requires_grad_(True)
    a = a_of(x0, y0).clone().requires_grad_(True)
    u = net(x, y, a)
    a_x, a_y = 2 * x0, 2 * y0
    a_xx = torch.full_like(x0, 2.0)

    u_x, _, u_a = total_grad(u, x, y, a, a_x, a_y)
    u_xx, _, _ = total_grad(u_x, x, y, a, a_x, a_y)
    u_xx = u_xx + u_a * a_xx

    err = (u_xx - gxx_true).abs().max().item()
    report("second total derivative d2u/dx2 (incl. the a_xx correction)",
           err < 1e-9, f"max |err| = {err:.3e}")


def check_grid_residual() -> None:
    """Manufactured solution: u = sin(pi x) sin(pi y), a = 1 + x^2 + y^2.

    The exact forcing is -div(a grad u), so the operator applied to (a, u) with
    f = 0 should return exactly that, and the gap should shrink like h^2.
    """
    print("         grid  |  max |err|   |  rate")
    prev_err = None
    ok = True
    for g in (33, 65, 129, 257):
        h = 1.0 / (g - 1)
        xs = torch.linspace(0, 1, g, dtype=DTYPE)
        ys = torch.linspace(0, 1, g, dtype=DTYPE)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        u = (torch.sin(PI * xx) * torch.sin(PI * yy))[None, None]
        a = (1.0 + xx ** 2 + yy ** 2)[None, None]

        # exact -div(a grad u) = -(a lap u + grad a . grad u)
        lap = -2 * PI ** 2 * torch.sin(PI * xx) * torch.sin(PI * yy)
        ux = PI * torch.cos(PI * xx) * torch.sin(PI * yy)
        uy = PI * torch.sin(PI * xx) * torch.cos(PI * yy)
        exact = -((1.0 + xx ** 2 + yy ** 2) * lap + 2 * xx * ux + 2 * yy * uy)
        exact = exact[None, None]

        # mask the border: the manufactured u is non-zero at the ghost ring, so
        # the Dirichlet zero-padding is deliberately wrong there
        got = darcy_residual_grid(u, a, h, f=0.0, scheme="conservative_arith")
        m = interior_mask(g, g, 1, dtype=DTYPE)
        err = ((got - exact).abs() * m).max().item()

        rate = "" if prev_err is None else f"{math.log2(prev_err / err):.2f}"
        print(f"         {g:4d}  |  {err:.4e}  |  {rate}")
        if prev_err is not None and math.log2(prev_err / err) < 1.8:
            ok = False
        prev_err = err

    report("conservative stencil converges at second order", ok,
           "each grid refinement should cut the error ~4x (rate ~2.0)")


def flux_x(x, y, a):
    """Exact qx = -a du/dx, written as an explicit function of (x, y, a).

    This mirrors how the mixed form actually works: the flux is a *separate
    network output*, differentiated once, not the derived expression -a*u_x.
    That distinction matters -- differentiating -a*u_x a second time would drag
    in d2a/dx2, which is exactly the term the mixed form exists to avoid.
    """
    s = torch.sin(PI * x) * torch.sin(PI * y)
    return -PI * torch.cos(PI * x) * torch.sin(PI * y) + s * (2 * x) / a


def flux_y(x, y, a):
    s = torch.sin(PI * x) * torch.sin(PI * y)
    return -PI * torch.sin(PI * x) * torch.cos(PI * y) + s * (2 * y) / a


def check_mixed_form() -> None:
    """Exact fluxes must annihilate all three residuals of the mixed system."""
    torch.manual_seed(0)
    n = 4096
    x0 = torch.rand(n, dtype=DTYPE).clamp(0.05, 0.95)
    y0 = torch.rand(n, dtype=DTYPE).clamp(0.05, 0.95)

    x = x0.clone().requires_grad_(True)
    y = y0.clone().requires_grad_(True)
    a = a_of(x0, y0).clone().requires_grad_(True)
    a_x, a_y = 2 * x0, 2 * y0

    u = net(x, y, a)
    qx = flux_x(x, y, a)                # Darcy's law satisfied by construction
    qy = flux_y(x, y, a)

    # the true forcing for this (a, u) pair, from the second-order form
    xc = x0.clone().requires_grad_(True)
    yc = y0.clone().requires_grad_(True)
    uc = composite(xc, yc)
    gx = torch.autograd.grad(uc, xc, torch.ones_like(uc), create_graph=True)[0]
    gy = torch.autograd.grad(uc, yc, torch.ones_like(uc), create_graph=True)[0]
    ac = a_of(xc, yc)
    fx = torch.autograd.grad(ac * gx, xc, torch.ones_like(gx), create_graph=True)[0]
    fy = torch.autograd.grad(ac * gy, yc, torch.ones_like(gy), create_graph=True)[0]
    f_true = -(fx + fy)

    r_x, r_y, r_m = darcy_residual_coords_mixed(u, qx, qy, x, y, a, a_x, a_y, f=0.0)
    # r_m was formed with f = 0, so it equals div q; div q - f_true should vanish
    e_law = max(r_x.abs().max().item(), r_y.abs().max().item())
    e_mass = (r_m - f_true).abs().max().item()

    report("mixed form: Darcy's law residual vanishes for exact fluxes",
           e_law < 1e-12, f"max |r_x|, |r_y| = {e_law:.3e}")
    report("mixed form: div q agrees with -div(a grad u)",
           e_mass < 1e-9, f"max |div q - f_true| = {e_mass:.3e}")


def check_against_reference_data() -> None:
    """The real test: does the operator reproduce PDEBench's own solver?

    Skipped silently when the prepared data is not present yet.
    """
    import h5py

    path = ROOT / "data" / "train" / "darcy_train.h5"
    if not path.exists():
        print("         (skipped -- run scripts/prepare_data.py first)")
        return

    with h5py.File(path, "r") as fh:
        a = torch.tensor(fh["a"][:32], dtype=DTYPE).unsqueeze(1)
        u = torch.tensor(fh["u"][:32], dtype=DTYPE).unsqueeze(1)
        xs = fh["x"][:]
        beta = float(fh.attrs.get("beta", 1.0))
    h = float(xs[1] - xs[0])

    r = darcy_residual_grid(u, a, h, f=beta, scheme="conservative_arith")
    rmse = r.pow(2).mean().sqrt().item()
    med = r.abs().median().item()
    ring = torch.cat([r[..., 0, :].flatten(), r[..., -1, :].flatten(),
                      r[..., 1:-1, 0].flatten(), r[..., 1:-1, -1].flatten()])
    ring_rmse = ring.pow(2).mean().sqrt().item()

    report("operator reproduces PDEBench's solver on ground-truth pairs",
           rmse < 0.15 and ring_rmse < 0.05,
           f"rmse {rmse:.3e} ({100 * rmse / beta:.2f} % of |f|), "
           f"median |r| {med:.3e}, boundary {ring_rmse:.3e}")

    # the boundary rule is the easiest thing to get wrong, so pin it explicitly
    from darcy.physics import _pad_replicate
    import darcy.physics as P
    saved = P._pad_dirichlet
    P._pad_dirichlet = _pad_replicate                    # simulate the wrong rule
    try:
        r_bad = darcy_residual_grid(u, a, h, f=beta, scheme="conservative_arith")
        ring_bad = torch.cat([r_bad[..., 0, :].flatten(), r_bad[..., -1, :].flatten(),
                              r_bad[..., 1:-1, 0].flatten(),
                              r_bad[..., 1:-1, -1].flatten()])
        bad = ring_bad.pow(2).mean().sqrt().item()
    finally:
        P._pad_dirichlet = saved
    report("zero-padding beats edge replication at the boundary",
           bad > 20 * ring_rmse,
           f"boundary rmse {ring_rmse:.3e} (zero-pad) vs {bad:.3e} (replicate) "
           f"-- {bad / ring_rmse:.0f}x worse")


def main() -> int:
    print("=" * 68)
    print("PHYSICS VERIFICATION")
    print("=" * 68)
    print("\n[1] first total derivatives")
    check_first_derivatives()
    print("\n[2] second total derivatives")
    check_second_derivatives()
    print("\n[3] grid residual, manufactured solution")
    check_grid_residual()
    print("\n[4] mixed first-order system")
    check_mixed_form()
    print("\n[5] against the reference PDEBench data")
    check_against_reference_data()
    print("\n" + "=" * 68)
    print(f"{'ALL CHECKS PASSED' if not _failures else f'{_failures} CHECK(S) FAILED'}")
    print("=" * 68)
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
