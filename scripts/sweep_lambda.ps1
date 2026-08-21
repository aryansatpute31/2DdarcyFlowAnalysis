<#
.SYNOPSIS
    Sweep physics.lambda_pde and report the accuracy / residual trade-off.

    The PDE residual divides by h^2, amplifying error ~16,000x, so lambda_pde
    cannot be guessed -- see results/lambda_sweep.md for what happens when you
    try. Re-run this after changing the residual, the grid, or the forcing.

.EXAMPLE
    .\scripts\sweep_lambda.ps1 -Arch fno
    .\scripts\sweep_lambda.ps1 -Arch fcn -Epochs 12 -NMax 4000
#>
[CmdletBinding()]
param(
    [ValidateSet("fno", "fcn")] [string] $Arch = "fno",
    [int] $Epochs = 8,
    [int] $NMax = 2000,
    [string[]] $Lambdas = @("1.0", "0.1", "0.01", "0.001", "0.0001", "0.00001"),
    [string] $OutRoot = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }
if (-not $OutRoot) { $OutRoot = Join-Path $Root "sweeps\$Arch" }

$base = @("out_root=$OutRoot", "optim.epochs=$Epochs", "data.n_train_max=$NMax",
          "log_every=0", "physics.warmup_epochs=2")

Write-Host "=== $Arch : lambda_pde sweep ($Epochs epochs, $NMax samples) ===" -ForegroundColor Cyan
Write-Host "baseline (no PDE)" -ForegroundColor Yellow
& $Python (Join-Path $Root "scripts\train.py") `
    --config (Join-Path $Root "configs\${Arch}_nopde.yaml") @base "name=sweep_base"

foreach ($lam in $Lambdas) {
    Write-Host "lambda_pde = $lam" -ForegroundColor Yellow
    & $Python (Join-Path $Root "scripts\train.py") `
        --config (Join-Path $Root "configs\${Arch}_pde.yaml") @base `
        "name=sweep_$lam" "physics.lambda_pde=$lam"
}

# ---- collect ---------------------------------------------------------------
$rows = @()
foreach ($d in Get-ChildItem (Join-Path $OutRoot "logs") -Directory) {
    $csv = Join-Path $d.FullName "history.csv"
    if (-not (Test-Path $csv)) { continue }
    $h = Import-Csv $csv
    $best = $h | Sort-Object { [double]$_.val_rel_l2 } | Select-Object -First 1
    $rows += [pscustomobject]@{
        lambda       = $d.Name -replace '^sweep_', ''
        val_rel_l2   = [double]$best.val_rel_l2
        val_residual = [double]$best.val_pde_residual
        boundary_err = [double]$best.val_boundary_rmse
    }
}

Write-Host "`n=== results (sorted by val rel-L2) ===" -ForegroundColor Cyan
$rows | Sort-Object val_rel_l2 | Format-Table `
    @{L = "lambda_pde"; E = { $_.lambda } },
    @{L = "val rel-L2"; E = { "{0:F5}" -f $_.val_rel_l2 } ; A = "right" },
    @{L = "val residual"; E = { "{0:E4}" -f $_.val_residual }; A = "right" },
    @{L = "boundary err"; E = { "{0:E3}" -f $_.boundary_err }; A = "right" } -AutoSize

Write-Host "Pick the largest lambda whose rel-L2 is still close to the baseline;"
Write-Host "that is where the residual improves most for the least accuracy cost."
