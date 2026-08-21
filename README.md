# 2D Darcy Flow — Forward Permeability → Pressure Surrogate

Neural surrogates for 2D Darcy flow on PDEBench: a Fourier neural operator and a
classic coordinate PINN, each trained with and without a PDE residual term —
measuring what a physics loss actually buys.

```
-div( a(x) grad u(x) ) = f      on  (0,1)^2
                  u(x) = 0      on  the boundary
```

Given the permeability / diffusion coefficient field `a(x)`, predict the pressure
(hydraulic head) field `u(x)`. Forcing is constant, `f = beta = 1.0`.

Two findings carry the study, and both are stated up front because they are the
point of the experiment rather than a by-product of it:

- **Architecture dominates.** The FNO is **13.4×** more accurate than the
  coordinate MLP on identical data and loss (0.0244 vs 0.3270 rel-L2), and a
  22-trial search spanning a 20× parameter range moves the MLP by **0.02 %**.
  That is a measured ceiling, not a tuning failure — a pointwise map can only
  learn `E[u | x, y, a(x,y)]`, while Darcy flow is elliptic and globally coupled.
- **The PDE term buys physical consistency, not accuracy.** For the FNO it costs
  2.9 % of relative L2 and returns a **2.7× lower** PDE residual plus a better
  boundary condition. For the coordinate MLP it changes nothing measurable.

Four models are trained and compared — two architectures crossed with the
presence or absence of a PDE residual term:

| run | architecture | loss |
|---|---|---|
| `fcn_nopde` | coordinate MLP (classic PINN) | data only |
| `fcn_pde`   | coordinate MLP (classic PINN) | data + PDE residual + BC |
| `fno_nopde` | Fourier neural operator | data only |
| `fno_pde`   | Fourier neural operator | data + PDE residual + BC |

Each was then re-tuned for accuracy, adding `fno_best`, `fno_pde_best` and
`fcn_best` — seven trained models in total.

> **Reading order.** This file documents the *code*: how to run it and how it is put
> together. For the study itself — results, derivations, analysis, limitations —
> read **[FINAL_REPORT.md](FINAL_REPORT.md)**. [REPORT.md](REPORT.md) is its
> auto-generated numbers-only counterpart, rebuilt by `scripts/make_report.py`.

---

## Dataset

**PDEBench**, `2D_DarcyFlow_beta1.0_Train.hdf5` — 10,000 samples on a 128×128
grid over the unit square.

- Source: DaRUS, `doi:10.18419/darus-2986`, datafile id `133219`
- Size: 1,310,724,488 bytes, MD5 `81694ed31306ff2e5f6b76349b0b4389`
- Citation: Takamoto et al., *PDEBench: An Extensive Benchmark for Scientific
  Machine Learning*, NeurIPS 2022 Datasets & Benchmarks

Raw keys are `nu` → the coefficient `a` and `tensor` → the solution `u`.

### What the data actually looks like

Three properties of this dataset are easy to assume wrongly, and each one
silently breaks the PDE term. `prepare_data.py` measures all three on every run
rather than trusting the assumption.

1. **The grid is cell-centred.** `x` runs from `h/2` to `1 - h/2` with
   `h = 1/128`. The Dirichlet wall is therefore *outside* the stored array: the
   true solution on the outermost stored ring is ≈0.014 RMSE, not zero. Solving
   the discrete equation at the boundary cells for the unknown ghost value
   recovers exactly **0** (to 1.5e-4 relative), so the residual stencil must
   **zero-pad** `u`. Replicating the edge instead inflates the boundary residual
   from 3.2e-03 to 41 — a factor of **13,000**.

2. **`a` is binary**, taking only the values `0.1` and `1.0`.

3. **Arithmetic face averaging matches the generating solver**, not harmonic.
   Harmonic averaging is the textbook choice for a discontinuous coefficient and
   genuinely wins on data generated that way — but on PDEBench it is ~50× worse
   at the `a`-jumps. This is measured, not assumed.

Ranking of every stencil × ghost-rule combination against the ground truth:

| scheme | RMSE | median &#124;r&#124; | boundary |
|---|---:|---:|---:|
| **conservative_arith / dirichlet** | **5.13e-02** | 9.52e-04 | **2.25e-03** |
| conservative (harmonic) / dirichlet | 2.50e+00 | 9.77e-04 | 3.36e+00 |
| expanded / interior-only | 4.07e+00 | 9.77e-04 | — |
| conservative_arith / neumann | 7.13e+00 | 9.77e-04 | 4.05e+01 |
| conservative (harmonic) / neumann | 7.44e+00 | 9.77e-04 | 4.00e+01 |

The winning row is the **discretisation noise floor**: no model can be expected
to drive the residual below ~5e-02 RMSE. `test.py` reports it alongside every
model's residual so the number is interpretable.

### Splits

```
2D_DarcyFlow_beta1.0_Train.hdf5   10,000 samples
        │
        ├── 95 %  →  data/train/darcy_train.h5      9,500
        │                    │
        │                    ├── 80 %  train fold   7,600   gradient steps
        │                    └── 20 %  validation   1,900   model selection only
        │
        └──  5 %  →  data/test/darcy_test.h5          500   held out entirely
```

The outer 95/5 split is a seeded permutation written once by `prepare_data.py`;
the exact indices land in `data/splits/split_manifest.json` so the split is
auditable and reproducible. The inner 80/20 split happens at run time in
`src/darcy/data.py`, seeded by `data.split_seed`.

**Normalisation statistics come from the 80 % training fold only** — never from
validation or test — so nothing leaks across a split boundary.

---

## Quick start

```powershell
# everything: download, split, train all four, evaluate, compare
.\scripts\run_all.ps1

# or step by step
.venv\Scripts\python.exe scripts\download_data.py            # 1.3 GB from DaRUS
.venv\Scripts\python.exe scripts\prepare_data.py             # 95/5 split
.venv\Scripts\python.exe scripts\verify_physics.py           # check the PDE code
.venv\Scripts\python.exe scripts\train.py --config configs\fno_pde.yaml
.venv\Scripts\python.exe scripts\test.py --run all
```

A five-epoch rehearsal of the whole pipeline:

```powershell
.\scripts\run_all.ps1 -Epochs 5 -Tag smoke
```

---

## Results

Held-out 5 % test split (500 samples, never seen in training or validation).
100 epochs each, seed 0, on an RTX 5060 Laptop.

| run | model | PDE | params | **test rel-L2** | median | RMSE | boundary err | PDE residual | ms/sample |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fno_nopde` | FNO | no | 1,188,353 | **0.0244** | 0.0094 | 7.77e-03 | 1.44e-03 | 10.83 | 0.78 |
| `fno_pde` | FNO | yes | 1,188,353 | **0.0251** | 0.0105 | 7.97e-03 | **1.26e-03** | **4.07** | 0.84 |
| `fcn_nopde` | coord. MLP | no | 66,689 | **0.3270** | 0.2781 | 7.51e-02 | 5.11e-03 | 227.2 | 0.62 |
| `fcn_pde` | coord. MLP | yes | 66,947 | **0.3273** | 0.2792 | 7.52e-02 | 7.04e-03 | 227.7 | 0.61 |

Discretisation noise floor (same operator on ground truth): **2.99e-02**.

### What the numbers say

**Architecture dominates.** The FNO is **13.4×** more accurate than the
coordinate MLP (0.0244 vs 0.3270) on identical data, and gets there with the
*same* inputs and loss. The gap is not a tuning artefact — it is the pointwise
map's ceiling. `N(x, y, a(x,y))` can only represent `E[u | x, y, a(x,y)]`, and
Darcy flow is elliptic, so `u` at a point depends on the *whole* field. Adding
parameters would not close this; changing the receptive field is what does.

**The PDE term buys physical consistency, not accuracy.** For the FNO it costs
2.9 % of relative L2 and returns a **2.7× lower PDE residual** (10.83 → 4.07)
plus a better boundary condition. That is a real, reproducible trade — worth
taking when the prediction feeds a downstream solver or a conservation check,
not worth taking if relative L2 is the only thing being scored.

**For the coordinate MLP it buys nothing.** rel-L2 0.3270 → 0.3273 and residual
227.2 → 227.7 are noise; the two runs also agree to six decimals on validation
(0.332253 vs 0.332251). The residual is dominated by the architecture's own
modelling error, which no weighting of a residual term can repair. Note this
also *contradicts the short sweep*, which showed the physics term halving the
coordinate MLP's boundary error — at full budget the data-only run reaches a
better boundary on its own (5.11e-03) and the physics run is worse (7.04e-03).
The short-budget advantage did not survive; reported as measured.

**No model approaches the noise floor.** The best residual (4.07) is still 136×
the floor (2.99e-02), so there is real headroom for a stronger physics
formulation — this is not a case of the metric being saturated.

Reproduce with `.\scripts\run_all.ps1`; per-sample numbers are in
`results/<run>/per_sample_metrics.csv` and field comparisons in
`results/<run>/figures/`.

---

## Hyperparameter search

Both architectures were then tuned for accuracy. 22 trials at a 20-epoch budget,
**ranked on the validation fold only** — the test split was not read until the
final models were trained. Leaderboards live in `search/<space>/leaderboard.csv`;
reproduce with `scripts/search.py --space fno|fcn`.

The two architectures gave opposite answers.

### FNO — 21.3 % better

| trial | val rel-L2 | params |
|---|---:|---:|
| **m20_w64_p8_L5_lr2e3** | **0.020687** | 16.4 M |
| m16_w64_p8_L5_lr2e3 | 0.020823 | 10.5 M |
| m16_w64_p8_lr2e3 | 0.021681 | 8.4 M |
| m16_w64_p8_L5 | 0.021732 | 10.5 M |
| m20_w64_p8 | 0.022492 | 13.1 M |
| m24_w64_p8 | 0.022538 | 18.9 M |
| m12_w32_p8 | 0.025773 | 1.2 M |
| base_m12_w32_p0 | 0.026271 | 1.2 M |

Three findings, in order of how much they mattered:

1. **Depth and learning rate each beat every capacity increase — and compound.**
   `n_layers=5` and `lr=2e-3` were the two best single changes, neither was in the
   original config, and the first grid never tested them together. Crossing them in
   a second round produced the winner. Worth remembering: the best configuration
   was not in the initial search space, and only appeared because round 1's results
   were used to design round 2.
2. **Spectral modes have a ceiling near 20.** Modes 24 scored *worse* than modes 20
   while carrying 44 % more parameters, so that lever is exhausted — useful to know
   before paying for a bigger model.
3. **The non-periodic padding fix is ~2 % for free.** The FFT treats the domain as
   periodic; Darcy with `u = 0` walls is not, and padding before the spectral stack
   costs no parameters at all (see `domain_padding` in `models/fno.py`).

### Coordinate PINN — 0.02 % better, i.e. untunable

| trial | val rel-L2 | params |
|---|---:|---:|
| **d6_w256_silu** | **0.332302** | 330 k |
| d6_w256_gelu | 0.332309 | 330 k |
| d8_w256_tanh | 0.332360 | 462 k |
| base_d5_w128_tanh | 0.332373 | 67 k |
| d6_w512_tanh | 0.332487 | 1.3 M |
| d6_w256_ff32 | 0.335926 | 346 k |
| d6_w256_ff64_s10 | 0.359203 | 362 k |

Ten trials spanning a **20× parameter range**, four activations, two coordinate
encodings, three learning rates and two collocation budgets. Total spread:
**0.0002**. This is not a failed search — it is a measurement of a hard ceiling,
and it converts the claim "a pointwise map cannot do this" from an assertion into
evidence.

The Fourier-feature trials are the most diagnostic result in the table. If the
bottleneck were spatial resolution, encoding `(x, y)` at higher frequency would
help; instead it made things monotonically *worse* (0.3359, then 0.3592). The
model is not failing to resolve fine detail — it is being asked to predict a
globally coupled quantity from purely local information, and no reparametrisation
of local inputs supplies the missing permeability field.

---

## The two architectures

### Coordinate MLP (`fcn`) — the classic PINN

A pointwise map evaluated independently at every collocation point:

```
N( x, y, a(x,y) )  ->  u(x,y)          [ + qx, qy in mixed mode ]
```

Spatial derivatives come from **autograd on the coordinate inputs**. Because the
network is `N(x, y, a(x,y))`, a spatial derivative is a *total* derivative and
picks up a chain-rule term:

```
du/dx = dN/dx + (dN/da) * (da/dx)
```

`da/dx` cannot come from autograd — `a` is tabulated data, not a closed form — so
it is supplied by a finite difference on the grid. `physics.total_grad`
implements exactly this operator, and `verify_physics.py` check 1 confirms that
dropping the chain-rule term would introduce an O(0.5) error.

> **Structural caveat, stated up front.** This is a *local* map: two different
> permeability fields sharing a value at one point get the same prediction there,
> so the network can only learn `E[u | x, y, a(x,y)]`. Darcy flow is elliptic and
> therefore globally coupled. The coordinate MLP is expected to trail the FNO by
> a wide margin — that gap is the headline result, not a bug. It is the reason
> neural *operators* exist.

### Fourier neural operator (`fno`)

Grid-to-grid, `a` field in → `u` field out. Each layer pairs a spectral
convolution (FFT → learned complex multiplier on the lowest `modes`
frequencies → inverse FFT) with a pointwise 1×1 convolution. Because the weights
live in frequency space rather than on grid nodes, the same checkpoint can be
evaluated on a different mesh.

Derivatives for the PDE term come from **finite-difference stencils** on the
predicted field — the FNO has no coordinate input to differentiate.

---

## The PDE term

`L = L_data + λ_pde · L_pde + λ_bc · L_bc`

- `L_data` — relative L2 error, the metric the FNO and PDEBench papers report.
  Scale free, so no target normalisation is needed.
- `L_pde` — squared PDE residual, normalised by `beta` so it is dimensionless.
- `L_bc` — squared `u` at the Dirichlet **ghost** ring, one spacing outside the
  cell-centred grid. Coordinate MLP only; the FNO stencil zero-pads, so the same
  condition is already inside its residual. Pinning the outermost *stored* ring
  to zero — the obvious thing to write — fights the labels, since the truth
  there is ≈0.014.

`λ_pde` ramps in linearly over `physics.warmup_epochs`. Without the ramp a large
residual term dominates from step zero and the model collapses to a trivial
field before learning anything from the data.

### λ_pde cannot be guessed

The residual divides by `h² `, amplifying prediction error ~16,000×, so `L_pde`
is far stiffer than the data term. A reasonable-looking `λ_pde = 0.1` makes the
FNO **15× worse** than using no physics at all. The shipped values come from a
sweep — 1e-4 for the FNO, 1e-3 for the coordinate MLP. See
[results/lambda_sweep.md](results/lambda_sweep.md) for the full curves, and
`scripts/sweep_lambda.ps1` to re-run it.

### Residual forms

**FNO — conservative five-point finite volume** with arithmetic face
coefficients and zero-padded `u`, the combination measured to match PDEBench's
own solver (see the table above).

**Coordinate MLP — mixed first-order system** (`physics.fcn_form: mixed`,
default). The network also emits the flux components, turning one second-order
equation into three first-order ones:

```
r_x = qx + a * du/dx           Darcy's law,  q = -a grad u
r_y = qy + a * du/dy
r_m = dqx/dx + dqy/dy - f      mass balance, div q = f
```

No second derivatives, and `a` is never differentiated inside a divergence.
That is what keeps it stable on rough coefficient fields. The textbook expanded
form is available as `fcn_form: second_order` — it is implemented correctly,
including the `d2a/dx2` correction term, but on piecewise-constant `a` its
initial residual is ~300× larger, since `d2a/dx2` scales like `jump / h^2`.

---

## Layout

```
configs/           one YAML per run; the only difference between the
                   *_pde and *_nopde pairs is physics.enabled
data/
  raw/             the downloaded PDEBench file
  train/ test/     the 95/5 split
  splits/          split_manifest.json — exact indices + diagnostics
src/darcy/
  config.py        typed config, YAML + `key=value` CLI overrides
  data.py          dataset, 80/20 split, normalisation statistics
  physics.py       both residual operators and the total-derivative machinery
  metrics.py       relative L2, RMSE, boundary error, residual
  models/fcn.py    coordinate MLP
  models/fno.py    Fourier neural operator
scripts/
  download_data.py resumable DaRUS download with MD5 verification
  prepare_data.py  95/5 split + physics sanity check
  verify_physics.py four correctness checks on the PDE code
  train.py         training entry point
  test.py          evaluation entry point
  h5_to_vtk.py     HDF5 fields -> VTK files for ParaView
  run_all.ps1      the whole pipeline
checkpoints/<run>/ best.pt, last.pt, config.yaml, model_summary.json
logs/<run>/        train.log, history.csv, metrics.json
results/<run>/     test_metrics.json, per_sample_metrics.csv, figures/
results/           comparison.{md,csv,json}, lambda_sweep.md
```

### What is and is not in the repository

The generated artefacts are large — ~2 GB of HDF5 under `data/`, 151 MB of
figures and predictions under `results/`, 14 MB of ParaView exports — and every
one of them is reproducible from the scripts above, so they are gitignored:

| path | tracked? | rebuild with |
|---|---|---|
| `data/raw/`, `data/train/`, `data/test/`, `data/splits/` | no | `download_data.py`, then `prepare_data.py` |
| `data/template.txt` | **yes** | — documents the expected `data/` layout |
| `logs/` | no | `train.py` |
| `paraview/` | no | `h5_to_vtk.py` |
| `results/<run>/` | no | `test.py` |
| `results/comparison.*`, `results/lambda_sweep.*` | **yes** | the summary tables |

A fresh clone therefore has an empty `data/` apart from
[`data/template.txt`](data/template.txt), which describes what each folder should
contain — the HDF5 keys, shapes, sample counts and the split arithmetic — so the
tree can be rebuilt and checked without downloading first.

Checkpoints are **self-contained**: weights, optimiser state, normalisation
buffers and the full config all travel inside the `.pt`. `test.py` rebuilds the
model straight from the checkpoint and does not depend on the config files still
being present or unchanged.

### Viewing the fields in ParaView

ParaView cannot open the `.h5` files directly — they are plain array dumps with
no mesh description attached. `scripts/h5_to_vtk.py` writes the same fields out
as VTK, one file per sample plus a `series.pvd` collection whose time slider is
the sample index:

```powershell
# ground truth + every model's prediction and error, first 20 test samples
python scripts/h5_to_vtk.py --input data/test/darcy_test.h5 `
    --pred results/fno_best --pred results/fcn_best --pred results/fno_pde_best `
    --samples 0:20 --out paraview/test_compare
```

Then open `paraview/test_compare/series.pvd`. Fields are `u` (truth), `a`
(permeability) and, per run, `u_pred_<run>`, `err_<run>`, `abs_err_<run>`.

Defaults to `.vti` (uniform image data — the grid *is* uniform, so this is both
the smallest and the fastest to render); `--format vtu` writes an unstructured
quad mesh and `--format vtk` the legacy binary form. It also reads the raw
PDEBench file (`nu` / `tensor`) and the train split. Needs only numpy + h5py.

---

## Metrics

`test.py` reports, per sample and aggregated (mean / median / p95 / max):

| metric | meaning |
|---|---|
| `rel_l2` | relative L2 error — **the headline number**, comparable to published results |
| `rmse`, `nrmse`, `mae`, `max_error` | absolute error measures |
| `boundary_rmse` | prediction error `\|pred − true\|` on the outermost ring — *not* `\|pred\|`, since the truth there is non-zero |
| `pde_residual_rmse` | how well the prediction satisfies the PDE |
| `pde_residual_rmse_truth` | the same operator on ground truth — the **discretisation noise floor** |

The residual is computed with the same finite-difference operator for every
architecture, so it is a common yardstick regardless of how a model was trained.
Reporting the ground-truth residual alongside it is what makes the number
interpretable: it is the floor no model can beat.

---

## Tuning notes

- `physics.lambda_pde` (default `0.1`) is the main knob. The finite-difference
  residual amplifies prediction error by `1/h² ≈ 16,000`, so it penalises
  high-frequency error hard — that is its regularising role, but it will
  overwhelm the data term if pushed too far.
- `model.points_per_sample` (default `4096`) trades coordinate-MLP cost against
  gradient noise.
- `model.fourier_features` > 0 switches on random Fourier encoding of `(x, y)`,
  which usually helps PINNs resolve high frequencies. Off by default to keep the
  baseline a textbook PINN.
- `optim.amp` is forced off whenever a PDE term is active; fp16 derivatives are
  not reliable inside a residual.
