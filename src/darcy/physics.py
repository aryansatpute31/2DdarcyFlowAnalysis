"""Darcy PDE residuals.

The governing equation on the unit square is

    -div( a(x) grad u(x) ) = f      in  (0,1)^2
                      u(x) = 0      on  the boundary

with a constant forcing f = beta.

Two residual operators live here, because the two architectures see the problem
through different eyes:

`darcy_residual_grid`
    For grid-to-grid models (the FNO).  The model has no coordinate input, so
    derivatives come from finite-difference stencils on the predicted field.

`darcy_residual_coords`
    For the coordinate MLP (the classic PINN).  Derivatives come from autograd.
    Because the network is N(x, y, a(x,y)), a spatial derivative is a *total*
    derivative and picks up a chain-rule term:

        du/dx = dN/dx + (dN/da) * (da/dx)

    da/dx is not available from autograd -- a is tabulated data, not a closed
    form -- so it is supplied by a finite difference on the grid.  `total_grad`
    below implements exactly that operator, and applying it twice (plus a
    d2a/dx2 correction) gives correct second derivatives.

Every function here expects *physical* (un-normalised) u and a.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Finite-difference helpers on a uniform grid
# ---------------------------------------------------------------------------
# Convention throughout: fields are [B, 1, H, W] with dim -2 indexing y and
# dim -1 indexing x, on a uniform grid of spacing h in both directions.


def _pad_replicate(t: torch.Tensor) -> torch.Tensor:
    """Edge replication -- the right ghost rule for the *coefficient* field."""
    return F.pad(t, (1, 1, 1, 1), mode="replicate")


def _pad_dirichlet(t: torch.Tensor) -> torch.Tensor:
    """Zero ghost ring -- the Dirichlet rule for the *solution* field.

    PDEBench's grid is cell-centred (x runs h/2 .. 1-h/2), so u = 0 does not
    hold at the outermost stored node; it holds one spacing further out.
    Solving the discrete equation at the boundary cells for the unknown ghost
    value recovers exactly 0 (to 1.5e-4 relative on the reference data), so
    zero-padding is what reproduces the generating solver.  Using replicate
    padding here instead inflates the boundary residual by four orders of
    magnitude and is the single easiest way to get this benchmark wrong.
    """
    return F.pad(t, (1, 1, 1, 1), mode="constant", value=0.0)


def _pad(field: torch.Tensor, mode: str) -> torch.Tensor:
    """`mode` is "replicate" for coefficient fields, "dirichlet" for solutions."""
    return _pad_dirichlet(field) if mode == "dirichlet" else _pad_replicate(field)


def grad_fd(field: torch.Tensor, h: float, pad: str = "replicate"
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Central first derivatives (d/dx, d/dy)."""
    p = _pad(field, pad)
    fx = (p[..., 1:-1, 2:] - p[..., 1:-1, :-2]) / (2.0 * h)
    fy = (p[..., 2:, 1:-1] - p[..., :-2, 1:-1]) / (2.0 * h)
    return fx, fy


def grad2_fd(field: torch.Tensor, h: float, pad: str = "replicate"
             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Central second derivatives (d2/dx2, d2/dy2)."""
    p = _pad(field, pad)
    c = p[..., 1:-1, 1:-1]
    fxx = (p[..., 1:-1, 2:] - 2.0 * c + p[..., 1:-1, :-2]) / (h * h)
    fyy = (p[..., 2:, 1:-1] - 2.0 * c + p[..., :-2, 1:-1]) / (h * h)
    return fxx, fyy


def laplacian_fd(field: torch.Tensor, h: float, pad: str = "replicate") -> torch.Tensor:
    fxx, fyy = grad2_fd(field, h, pad)
    return fxx + fyy


def interior_mask(h_pts: int, w_pts: int, width: int = 1,
                  device=None, dtype=torch.float32) -> torch.Tensor:
    """[1, 1, H, W] mask that is 1 on interior nodes and 0 on a border ring."""
    m = torch.zeros(1, 1, h_pts, w_pts, device=device, dtype=dtype)
    m[..., width:h_pts - width, width:w_pts - width] = 1.0
    return m


# ---------------------------------------------------------------------------
# Grid residual (FNO)
# ---------------------------------------------------------------------------

def darcy_residual_grid(u: torch.Tensor, a: torch.Tensor, h: float,
                        f: float = 1.0, scheme: str = "conservative_arith"
                        ) -> torch.Tensor:
    """Residual r = -div(a grad u) - f evaluated on the grid.

    The solution is zero-padded (Dirichlet) and the coefficient edge-replicated,
    which together reproduce PDEBench's generating solver -- see `_pad_dirichlet`.
    Because the boundary condition is baked into the padding, the residual is
    meaningful on the **whole** field, boundary ring included; no masking needed.

    Parameters
    ----------
    u, a : [B, 1, H, W] physical fields.
    h    : grid spacing.
    f    : constant forcing (PDEBench beta).
    scheme :
        "conservative_arith" five-point finite volume, arithmetic face
                            coefficients.  This is the default because it is
                            what matches the reference data -- measured, not
                            assumed; `prepare_data.py` re-checks it every run.
        "conservative"      same, harmonic face coefficients.  Textbook advice
                            for a discontinuous coefficient, and it does beat
                            arithmetic on data generated that way, but on
                            PDEBench it is ~50x worse at the a-jumps.
        "expanded"          -(a * lap(u) + grad(a) . grad(u)) - f.  Assumes a is
                            differentiable, so it is noisier on rough a.

    Returns
    -------
    [B, 1, H, W] residual.
    """
    if scheme in ("conservative", "conservative_arith"):
        harmonic = scheme == "conservative"
        up = _pad_dirichlet(u)
        ap = _pad_replicate(a)

        ac = ap[..., 1:-1, 1:-1]
        a_e, a_w = ap[..., 1:-1, 2:], ap[..., 1:-1, :-2]
        a_n, a_s = ap[..., 2:, 1:-1], ap[..., :-2, 1:-1]

        uc = up[..., 1:-1, 1:-1]
        u_e, u_w = up[..., 1:-1, 2:], up[..., 1:-1, :-2]
        u_n, u_s = up[..., 2:, 1:-1], up[..., :-2, 1:-1]

        if harmonic:
            eps = 1e-12
            face = lambda p, q: 2.0 * p * q / (p + q + eps)  # noqa: E731
        else:
            face = lambda p, q: 0.5 * (p + q)                # noqa: E731

        div = (face(ac, a_e) * (u_e - uc) - face(ac, a_w) * (uc - u_w)
               + face(ac, a_n) * (u_n - uc) - face(ac, a_s) * (uc - u_s)) / (h * h)
        return -div - f

    if scheme == "expanded":
        ux, uy = grad_fd(u, h, pad="dirichlet")
        ax, ay = grad_fd(a, h, pad="replicate")
        lap = laplacian_fd(u, h, pad="dirichlet")
        return -(a * lap + ax * ux + ay * uy) - f

    raise ValueError(f"unknown fd scheme '{scheme}'")


# ---------------------------------------------------------------------------
# Coordinate residual (classic PINN)
# ---------------------------------------------------------------------------

def total_grad(out: torch.Tensor, x: torch.Tensor, y: torch.Tensor, a: torch.Tensor,
               a_x: torch.Tensor, a_y: torch.Tensor, create_graph: bool = True
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Total spatial derivatives of a network output N(x, y, a(x,y)).

    Returns (d/dx, d/dy, dN/da) where

        d out /dx = dN/dx + (dN/da) * a_x
        d out /dy = dN/dy + (dN/da) * a_y

    `a_x`, `a_y` are the finite-difference derivatives of the tabulated
    coefficient field at the same points.  dN/da is returned as well because the
    second-derivative correction term needs it.
    """
    ones = torch.ones_like(out)
    g_x, g_y, g_a = torch.autograd.grad(
        out, (x, y, a), grad_outputs=ones,
        create_graph=create_graph, retain_graph=True,
    )
    return g_x + g_a * a_x, g_y + g_a * a_y, g_a


def darcy_residual_coords_mixed(
    u: torch.Tensor, qx: torch.Tensor, qy: torch.Tensor,
    x: torch.Tensor, y: torch.Tensor, a: torch.Tensor,
    a_x: torch.Tensor, a_y: torch.Tensor, f: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """First-order (mixed) form of Darcy flow.

    The network predicts the pressure and both flux components, turning one
    second-order equation into three first-order ones:

        r_x = qx + a * du/dx        Darcy's law,  q = -a grad u
        r_y = qy + a * du/dy
        r_m = dqx/dx + dqy/dy - f   mass balance, div q = -div(a grad u) = f

    No second derivatives and no derivative of `a` inside a divergence.  That is
    what makes it usable on PDEBench, whose a(x) is binary -- taking only 0.1 and
    1.0 -- so d2a/dx2 at a jump scales like 1/h^2 and would swamp everything else
    in the second-order form.
    """
    u_x, u_y, _ = total_grad(u, x, y, a, a_x, a_y)
    qx_x, _, _ = total_grad(qx, x, y, a, a_x, a_y)
    _, qy_y, _ = total_grad(qy, x, y, a, a_x, a_y)

    r_x = qx + a * u_x
    r_y = qy + a * u_y
    r_m = qx_x + qy_y - f
    return r_x, r_y, r_m


def darcy_residual_coords_second_order(
    u: torch.Tensor, x: torch.Tensor, y: torch.Tensor, a: torch.Tensor,
    a_x: torch.Tensor, a_y: torch.Tensor,
    a_xx: torch.Tensor, a_yy: torch.Tensor, f: float = 1.0,
) -> torch.Tensor:
    """Textbook expanded form: r = -(a * lap(u) + grad(a) . grad(u)) - f.

    Second total derivatives follow from applying `total_grad` twice and adding
    the term the chain rule leaves behind:

        d2u/dx2 = total_d/dx( du/dx ) + (dN/da) * a_xx
    """
    u_x, u_y, u_a = total_grad(u, x, y, a, a_x, a_y)
    u_xx, _, _ = total_grad(u_x, x, y, a, a_x, a_y)
    _, u_yy, _ = total_grad(u_y, x, y, a, a_x, a_y)
    u_xx = u_xx + u_a * a_xx
    u_yy = u_yy + u_a * a_yy

    return -(a * (u_xx + u_yy) + a_x * u_x + a_y * u_y) - f


# ---------------------------------------------------------------------------
# Sampling helpers for the coordinate model
# ---------------------------------------------------------------------------

def coefficient_derivative_fields(a: torch.Tensor, h: float
                                  ) -> dict[str, torch.Tensor]:
    """Precompute a_x, a_y, a_xx, a_yy for a batch of coefficient fields."""
    a_x, a_y = grad_fd(a, h)
    a_xx, a_yy = grad2_fd(a, h)
    return {"a_x": a_x, "a_y": a_y, "a_xx": a_xx, "a_yy": a_yy}


def gather_at(field: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Sample [B, 1, H, W] fields at flat node indices [B, P] -> [B, P]."""
    b, c, hgt, wid = field.shape
    assert c == 1, "gather_at expects single-channel fields"
    return field.reshape(b, hgt * wid).gather(1, idx)


def sample_interior_indices(batch: int, h_pts: int, w_pts: int, n_points: int,
                            device, generator: torch.Generator | None = None
                            ) -> torch.Tensor:
    """Random interior node indices, [B, n_points], excluding the border ring.

    The border is excluded because the finite-difference a_x there is one-sided
    and because those nodes are already pinned by the Dirichlet term.
    """
    ii = torch.randint(1, h_pts - 1, (batch, n_points), device=device,
                       generator=generator)
    jj = torch.randint(1, w_pts - 1, (batch, n_points), device=device,
                       generator=generator)
    return ii * w_pts + jj


def ghost_boundary_samples(batch: int, h_pts: int, w_pts: int, n_points: int,
                           xs: torch.Tensor, ys: torch.Tensor, h: float, device,
                           generator: torch.Generator | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample the Dirichlet ghost points, one spacing outside the stored grid.

    The grid is cell-centred, so u = 0 holds *outside* the stored array, not on
    its outermost ring -- on the reference data the ring carries u ~ 0.014 while
    the ghost value is 0 to within 1.5e-4 relative.  Pinning the stored ring to
    zero would therefore fight the labels rather than help them.

    Returns
    -------
    (idx_adjacent, x_ghost, y_ghost)
        `idx_adjacent` is the flat index of the in-domain cell next to each ghost
        point; look up the coefficient there, matching the edge-replicated `a`
        ghost used by the grid stencil.
    """
    side = torch.randint(0, 4, (batch, n_points), device=device, generator=generator)
    ti = torch.randint(0, h_pts, (batch, n_points), device=device, generator=generator)
    tj = torch.randint(0, w_pts, (batch, n_points), device=device, generator=generator)

    zero_i, zero_j = torch.zeros_like(ti), torch.zeros_like(tj)
    ii = torch.where(side == 0, zero_i,
                     torch.where(side == 1, torch.full_like(ti, h_pts - 1), ti))
    jj = torch.where(side == 2, zero_j,
                     torch.where(side == 3, torch.full_like(tj, w_pts - 1), tj))

    x = xs[jj]
    y = ys[ii]
    zero = torch.zeros_like(x)
    step = torch.full_like(x, h)
    dx = torch.where(side == 2, -step, torch.where(side == 3, step, zero))
    dy = torch.where(side == 0, -step, torch.where(side == 1, step, zero))

    return ii * w_pts + jj, x + dx, y + dy


def node_coordinates(idx: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor,
                     w_pts: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn flat node indices into physical (x, y) coordinates."""
    ii = torch.div(idx, w_pts, rounding_mode="floor")
    jj = idx % w_pts
    return xs[jj], ys[ii]
