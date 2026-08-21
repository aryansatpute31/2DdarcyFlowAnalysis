<#
.SYNOPSIS
    End-to-end pipeline: prepare data, train all four models, evaluate and compare.

.EXAMPLE
    .\scripts\run_all.ps1
    .\scripts\run_all.ps1 -Epochs 5 -Tag smoke      # quick end-to-end rehearsal
    .\scripts\run_all.ps1 -Only fno_pde,fcn_pde     # just the physics-informed runs
    .\scripts\run_all.ps1 -SkipPrepare              # data already split
#>
[CmdletBinding()]
param(
    [int]    $Epochs = 0,                                # 0 -> use each config's value
    [string] $Tag = "",                                  # suffix appended to run names
    [string[]] $Only = @("fcn_nopde", "fcn_pde", "fno_nopde", "fno_pde"),
    [switch] $SkipPrepare,
    [switch] $SkipTrain,
    [switch] $SavePredictions,
    # Continue any run that already has a checkpoints/<name>/last.pt, instead of
    # restarting it from scratch. Runs with no checkpoint still start fresh.
    [switch] $Resume,
    # Long GPU runs occasionally die on a transient driver fault (we hit a
    # `CUDA error: unknown error` mid-backward after 46 clean epochs). Every
    # epoch writes last.pt, so a retry with --resume picks up where it stopped
    # instead of throwing away hours of training.
    [int]    $Retries = 3
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

# `powershell -File script.ps1 -Only a,b` hands the whole thing over as ONE
# string rather than a two-element array, so split it back apart. Costs nothing
# when the caller already passed a proper array.
$Only = $Only | ForEach-Object { $_ -split ',' } | Where-Object { $_ -ne '' }

# Fail fast on a typo instead of discovering it three retries deep.
foreach ($cfg in $Only) {
    $p = Join-Path $Root "configs\$cfg.yaml"
    if (-not (Test-Path $p)) {
        $avail = (Get-ChildItem (Join-Path $Root "configs") -Filter *.yaml |
                  ForEach-Object { $_.BaseName }) -join ", "
        throw "no config '$cfg' (looked for $p). Available: $avail"
    }
}

Write-Host "`n=== Darcy flow surrogate pipeline ===" -ForegroundColor Cyan
Write-Host "root   : $Root"
Write-Host "python : $Python`n"

# ---- 0. verify the PDE code ------------------------------------------------
Write-Host "[0/3] verifying physics operators" -ForegroundColor Yellow
& $Python (Join-Path $Root "scripts\verify_physics.py")
if ($LASTEXITCODE -ne 0) { throw "physics verification failed -- stopping" }

# ---- 1. data ---------------------------------------------------------------
$TrainFile = Join-Path $Root "data\train\darcy_train.h5"
if (-not $SkipPrepare -and -not (Test-Path $TrainFile)) {
    $Raw = Join-Path $Root "data\raw\2D_DarcyFlow_beta1.0_Train.hdf5"
    if (-not (Test-Path $Raw)) {
        Write-Host "[1/3] downloading PDEBench Darcy (1.3 GB)" -ForegroundColor Yellow
        & $Python (Join-Path $Root "scripts\download_data.py") --beta 1.0
        if ($LASTEXITCODE -ne 0) { throw "download failed" }
    }
    Write-Host "[1/3] splitting 95/5 into train/test" -ForegroundColor Yellow
    & $Python (Join-Path $Root "scripts\prepare_data.py")
    if ($LASTEXITCODE -ne 0) { throw "prepare_data failed" }
} else {
    Write-Host "[1/3] data already prepared, skipping" -ForegroundColor DarkGray
}

# ---- 2. training -----------------------------------------------------------
$RunNames = @()
if (-not $SkipTrain) {
    foreach ($cfg in $Only) {
        $name = if ($Tag) { "${cfg}_$Tag" } else { $cfg }
        $RunNames += $name
        Write-Host "`n[2/3] training $name" -ForegroundColor Yellow

        # [string[]] is load-bearing on every splatted variable below. PowerShell
        # unwraps a single-element array to a scalar, and splatting a *string*
        # iterates its characters -- so a bare @("--resume") coming out of an
        # `if` expression reaches python as '-','-','r','e','s','u','m','e'.
        [string[]]$ovr = @("name=$name")
        if ($Epochs -gt 0) { $ovr += "optim.epochs=$Epochs" }

        $started = Get-Date
        $ckpt = Join-Path $Root "checkpoints\$name\last.pt"
        $hasCkpt = Test-Path $ckpt
        $lastStamp = if ($hasCkpt) { (Get-Item $ckpt).LastWriteTimeUtc } else { [datetime]::MinValue }

        for ($try = 0; $try -le $Retries; $try++) {
            [string[]]$extra = @()
            if ($try -gt 0 -or ($Resume -and $hasCkpt)) { $extra = @("--resume") }
            if ($try -gt 0) {
                Write-Host "  retry $try/$Retries -- resuming from last.pt" -ForegroundColor Magenta
                Start-Sleep -Seconds 20        # let the driver settle
            }
            & $Python (Join-Path $Root "scripts\train.py") `
                --config (Join-Path $Root "configs\$cfg.yaml") @extra @ovr
            if ($LASTEXITCODE -eq 0) { break }

            Write-Host "  training died (exit $LASTEXITCODE)" -ForegroundColor Red
            if ($try -eq $Retries) { throw "training failed for $cfg after $Retries retries" }

            # Only retry if the attempt actually got somewhere. A transient GPU
            # fault leaves a freshly written last.pt behind; a deterministic
            # failure (bad config, shape bug) writes nothing, and retrying it
            # just burns time and hides the real error.
            $stamp = if (Test-Path $ckpt) { (Get-Item $ckpt).LastWriteTimeUtc } else { [datetime]::MinValue }
            if ($stamp -le $lastStamp) {
                throw ("training for $cfg failed without completing an epoch -- " +
                       "this looks deterministic, not a transient. See the traceback above.")
            }
            $lastStamp = $stamp
        }
        Write-Host ("  finished in {0:hh\:mm\:ss}" -f ((Get-Date) - $started)) -ForegroundColor DarkGray
    }
} else {
    $RunNames = $Only | ForEach-Object { if ($Tag) { "${_}_$Tag" } else { $_ } }
    Write-Host "[2/3] training skipped" -ForegroundColor DarkGray
}

# ---- 3. evaluation ---------------------------------------------------------
Write-Host "`n[3/3] evaluating on the held-out test split" -ForegroundColor Yellow
[string[]]$ckpts = @($RunNames | ForEach-Object { Join-Path $Root "checkpoints\$_\best.pt" } |
                     Where-Object { Test-Path $_ })
if (-not $ckpts) { throw "no checkpoints found to evaluate" }

[string[]]$testArgs = @("--checkpoint") + $ckpts
if ($SavePredictions) { $testArgs += "--save-predictions" }
& $Python (Join-Path $Root "scripts\test.py") @testArgs
if ($LASTEXITCODE -ne 0) { throw "testing failed" }

Write-Host "`n=== done ===" -ForegroundColor Cyan
Write-Host "comparison table : $(Join-Path $Root 'results\comparison.md')"
Write-Host "per-run results  : $(Join-Path $Root 'results\<run>\')"
Write-Host "training logs    : $(Join-Path $Root 'logs\<run>\')"
