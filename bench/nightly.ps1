# Full Phase 1 + Phase 2 measurement sweep. Meant to run unattended overnight.
#
# Records what else was running, so a polluted result is identifiable after the
# fact instead of being quietly wrong. Never closes anything: if the machine is
# busy the numbers are still recorded, just annotated as busy.

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outdir = Join-Path $root "bench\runs\$stamp"
New-Item -ItemType Directory -Force -Path $outdir | Out-Null
$log = Join-Path $outdir "run.log"

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Output $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

Log "vkgrad nightly sweep starting"
Log "host: $env:COMPUTERNAME  user: $env:USERNAME"

# Machine-quiet evidence. 60 samples of total CPU over ~60s.
$cpu = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 15 |
        ForEach-Object { $_.CounterSamples[0].CookedValue } | Measure-Object -Average).Average
Log ("baseline CPU load: {0:N1}%" -f $cpu)
if ($cpu -gt 15) { Log "WARNING: machine is busy, bandwidth numbers will be understated" }

$mem = Get-CimInstance Win32_OperatingSystem
Log ("free RAM: {0:N1} GiB of {1:N1} GiB" -f ($mem.FreePhysicalMemory / 1MB), ($mem.TotalVisibleMemorySize / 1MB))

Get-Process | Sort-Object -Descending CPU | Select-Object -First 15 Name, CPU, WorkingSet |
    Format-Table -AutoSize | Out-File -FilePath (Join-Path $outdir "processes.txt") -Encoding utf8
Log "top processes recorded"

Set-Location $root
$py = "C:\Python314\python.exe"

Log "=== [1] roofline sweep (full) ==="
& $py -m bench.roofline --json (Join-Path $outdir "roofline.json") 2>&1 | Out-File -FilePath (Join-Path $outdir "roofline.txt") -Append -Encoding utf8
Log "roofline exit: $LASTEXITCODE"

# Added by Phase 2; skipped cleanly until it exists.
if (Test-Path (Join-Path $root "bench\matmul_sweep.py")) {
    Log "=== [2] matmul roofline sweep ==="
    & $py -m bench.matmul_sweep --json (Join-Path $outdir "matmul.json") 2>&1 | Out-File -FilePath (Join-Path $outdir "matmul.txt") -Append -Encoding utf8
    Log "matmul exit: $LASTEXITCODE"
} else {
    Log "matmul sweep not present yet, skipped"
}

Log "=== [3] end-to-end training step ==="
$mnistReady = Test-Path (Join-Path $root "data\train-images-idx3-ubyte.gz")
$dataFlag = if ($mnistReady) { @() } else { @("--synthetic") }
if (-not $mnistReady) { Log "no MNIST data present, using synthetic input" }
& $py -m examples.mnist @dataFlag --epochs 3 2>&1 | Out-File -FilePath (Join-Path $outdir "train.txt") -Append -Encoding utf8
Log "training exit: $LASTEXITCODE"

Log "=== [4] transformer training step ==="
& $py -m examples.charlm --steps 300 --tune 2>&1 | Out-File -FilePath (Join-Path $outdir "charlm.txt") -Append -Encoding utf8
Log "charlm exit: $LASTEXITCODE"

$cpu2 = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 5 |
         ForEach-Object { $_.CounterSamples[0].CookedValue } | Measure-Object -Average).Average
Log ("closing CPU load: {0:N1}%" -f $cpu2)
Log "done. results in $outdir"
