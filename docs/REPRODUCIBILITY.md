# Reproducibility guide

## Environment

- Python 3.11 or newer
- Windows PowerShell for the provided orchestration scripts
- Sufficient disk space for the 4.76 GB compressed Nasdaq sample and local
  Parquet output

Create the environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup.ps1
.venv\Scripts\Activate.ps1
```

Validate the installation:

```powershell
python -m ruff check src tests
python -m ruff format --check src tests
pytest
```

PyArrow is required for the real streaming-Parquet test. Without it, that test
is skipped.

## Predictive-data setup

Download the public AAPL LOBSTER sample:

```powershell
.\scripts\download_lobster.ps1
```

Expected local files:

```text
data/raw/lobster/AAPL_10/
  AAPL_2012-06-21_34200000_57600000_message_10.csv
  AAPL_2012-06-21_34200000_57600000_orderbook_10.csv
```

Run the predictive stages:

```powershell
.\scripts\run_phase0.ps1
.\scripts\run_all_baselines.ps1
.\scripts\run_walk_forward.ps1
.\scripts\run_bootstrap.ps1
.\scripts\run_diagnostics.ps1
.\scripts\run_phase_b.ps1
.\scripts\run_phase_c.ps1
```

The final candidate protocol is frozen in
`docs/FINAL_EVALUATION_PROTOCOL.md`. The final runner writes a manifest with the
Git commit, configuration hash, Python version, and completion time. It refuses
an accidental second final run unless an explicit override is supplied.

## Raw ITCH setup

Run the synthetic lifecycle test before downloading the large file:

```powershell
.\scripts\run_phase_e.ps1 -Smoke
```

Download the official sample:

```powershell
.\scripts\download_itch_sample.ps1 -ConfirmLargeDownload
```

The downloader:

- reuses a complete existing file;
- verifies the expected compressed byte size;
- verifies MD5 when the Nasdaq checksum endpoint is available;
- reports a warning, rather than a false download failure, when the checksum
  endpoint is unavailable.

Run a limited AAPL reconstruction:

```powershell
.\scripts\run_phase_e.ps1 `
    -InputPath "data\raw\itch\01302019.NASDAQ_ITCH50.gz" `
    -Symbol AAPL `
    -Levels 10 `
    -StopAfterTargetEvents 250000
```

Inspect:

```powershell
$result = Get-Content `
    reports\tables\phase_e_itch_AAPL_reconstruction_metrics.json `
    -Raw | ConvertFrom-Json

$result.reconstruction | Format-List
$result.final_integrity | Format-List
$result.benchmark | Format-List
```

For a deliberate limited run, `stopped_early` is true and `end_marker_seen` is
false. All order-reference, quantity, timestamp, aggregation, and crossed-book
failure counters should remain zero.

Run the complete session only after the limited run passes:

```powershell
.\scripts\run_phase_e.ps1 `
    -InputPath "data\raw\itch\01302019.NASDAQ_ITCH50.gz" `
    -Symbol AAPL `
    -Levels 10
```

The reference full run took about 2.5 hours on the development machine. Runtime
depends on CPU, storage, decompression throughput, antivirus scanning, and
Python/PyArrow versions.

## Output policy

Public Git contains:

- source code and tests;
- aggregate metrics and fold summaries;
- final figures;
- executed notebooks;
- protocols and documentation.

Public Git excludes:

- raw market data;
- Parquet reconstruction output;
- trained model binaries;
- smoke-test outputs;
- event-level derived samples;
- row-level holdout simulations.

## Determinism and interpretation

The linear and LightGBM builders use fixed parameters and seeds. Small numerical
variation may still occur across operating systems, compiler libraries, and
package versions. Reproduced conclusions should be assessed from the direction
and scale of the metrics, not from byte-for-byte model equality.
