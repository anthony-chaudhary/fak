# Cross-Machine Benchmark Infrastructure Design

**Status:** Design v1.0 | **Last Updated:** 2025-01-06

## Overview

This infrastructure enables scalable, queryable, and visualizable benchmark results across heterogeneous machines (Mac M3 Pro, Windows RTX 4070/WSL2, Desktop, A100s, Blackwell, and future hardware).

The design builds on existing patterns:
- Phase 0 node benchmark structure (`fak/experiments/fleet-nodes/`)
- Model baseline comparisons (`fak/experiments/model-baseline/`)
<!-- GPU server endpoint-load artifacts excluded from the public copy (operator-private lab infra). -->

## 1. Storage Structure

### Primary Layout: `experiments/benchmark/`

```
experiments/benchmark/
├── catalog.json                          # Master index of all runs
├── machines/                             # Machine registry and specs
│   ├── catalog.json                      # Machine inventory
│   ├── anthony-laptop/
│   │   ├── specs.json                     # One-time machine specs
│   │   └── history/                      # Optional time-series specs
│   └── mac-m3pro/
│       └── specs.json
├── runs/                                 # All benchmark runs
│   ├── by-machine/                       # Primary: machine-first view
│   │   ├── anthony-laptop/
│   │   │   ├── 20250106-120000-rtx4070-q8/
│   │   │   │   ├── manifest.json         # Run metadata
│   │   │   │   ├── kernel.json           # Core kernel results
│   │   │   │   ├── batch.json            # Batched decode curve
│   │   │   │   └── fleetbench.json       # Fleet turn-tax surface
│   │   │   └── 20250106-140000-rtx4070-int8/
│   │   └── mac-m3pro/
│   │       └── 20250106-103000-m3-q8/
│   ├── by-model/                          # Secondary: model-first view (symlinks or index)
│   │   └── smollm2-135m/
│   │       ├── anthony-laptop-rtx4070-q8.json
│   │       └── mac-m3pro-q8.json
│   └── by-date/                           # Tertiary: chronological view
│       └── 2025-01-06/
│           ├── anthony-laptop-rtx4070-q8.json
│           └── mac-m3pro-q8.json
└── charts/                               # Generated visualizations
    ├── throughput-vs-machine.html
    ├── scaling-curves.html
    └── cost-per-token.html
```

### Naming Convention

Run directories: `<machine-id>-<model-id>-<precision>-<config-hash>-<timestamp>`

Example: `anthony-laptop-smollm2-135m-q8-batch-a1b2c3d-20250106T120000Z`

Components:
- `<machine-id>`: Slug from machines catalog (e.g., `anthony-laptop`, `mac-m3pro`)
- `<model-id>`: Model slug (e.g., `smollm2-135m`, `qwen3-122b`, `deepseek-r1-70b`)
- `<precision>`: Quantization (e.g., `f32`, `q8`, `q4`, `bf16`)
- `<config-hash>`: 8-char hash of config JSON for deduplication
- `<timestamp>`: ISO 8601 UTC (YYYYMMDDThhmmssZ)

### File Names Within Runs

Each run contains:

```
<run-id>/
├── manifest.json           # REQUIRED: Run metadata (see schema)
├── kernel.json             # REQUIRED: Kernel benchmark results
├── batch.json              # REQUIRED: Batched decode results
├── modelbench.json         # OPTIONAL: Single-stream model benchmark
├── fleetbench.json         # OPTIONAL: Fleet turn-tax surface
├── workload.json           # OPTIONAL: Workload trace used
├── gate.json               # OPTIONAL: Gate validation result
└── README.md               # OPTIONAL: Human-readable notes
```

## 2. Metadata Schema

### Machine Specs (`machines/<machine-id>/specs.json`)

```json
{
  "$schema": "benchmark/machine-specs.v1",
  "machine_id": "anthony-laptop",
  "hostname": "ANTHONY-WIN",
  "registered_at": "2025-01-06T10:00:00Z",
  "hardware": {
    "cpu": {
      "model": "Intel64 Family 6 Model 186",
      "cores_physical": 6,
      "cores_logical": 20,
      "frequency_base_mhz": 2300,
      "architecture": "x86_64"
    },
    "gpu": {
      "model": "NVIDIA RTX 4070",
      "memory_gb": 12,
      "compute_capability": "8.9",
      "cuda_version": "12.6"
    },
    "ram_gb": 64,
    "storage": {
      "type": "NVMe SSD",
      "size_gb": 1024
    }
  },
  "os": {
    "name": "Windows 11 Pro",
    "version": "10.0.26200",
    "kernel": "Windows NT"
  },
  "runtime": {
    "go_version": "go1.26.0 windows/amd64",
    "python_version": "3.11.8",
    "cuda_driver_version": "560.94"
  },
  "tags": ["amd64", "cuda", "wsl2", "laptop"]
}
```

### Run Manifest (`runs/by-machine/<machine>/<run-id>/manifest.json`)

```json
{
  "$schema": "benchmark/run-manifest.v1",
  "run_id": "anthony-laptop-smollm2-135m-q8-batch-a1b2c3d-20250106T120000Z",
  "machine_id": "anthony-laptop",
  "timestamp": "2025-01-06T12:00:00Z",
  "git": {
    "rev": "9a0135c",
    "branch": "master",
    "dirty": false
  },
  "harness": {
    "name": "fak",
    "version": "0.24.0"
  },
  "model": {
    "name": "SmolLM2-135M-Instruct",
    "family": "smollm2",
    "parameters": "135M",
    "source": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "precision": "q8",
    "quantization": "Q8_0"
  },
  "config": {
    "batch_sizes": [1, 4, 8, 16, 32, 64, 128, 256, 512],
    "workers": 16,
    "decode_steps": 32,
    "prefill_sizes": [16, 64, 256],
    "workload_file": "fak/experiments/agent-live/production-workload.json",
    "workload_prefill_cap": 0,
    "workload_prompt_cap": 0
  },
  "tags": ["kernel", "batch", "smoke"],
  "artifacts": {
    "kernel": "kernel.json",
    "batch": "batch.json",
    "fleetbench": "fleetbench.json"
  },
  "gate": {
    "required": true,
    "status": "passed",
    "min_speedup": 45.0,
    "actual_speedup": 44.92
  }
}
```

### Kernel Results (`kernel.json`)

```json
{
  "$schema": "benchmark/kernel-results.v1",
  "run_id": "anthony-laptop-smollm2-135m-q8-batch-a1b2c3d-20250106T120000Z",
  "engine": "fak Q8_0 (all-core)",
  "precision": "Q8_0",
  "model": "SmolLM2-135M",
  "benchmark": {
    "prefill": [
      {"tokens": 16, "median_ms": 8.2, "tok_per_sec": 1951.2},
      {"tokens": 64, "median_ms": 28.5, "tok_per_sec": 2245.6},
      {"tokens": 256, "median_ms": 145.8, "tok_per_sec": 1756.0}
    ],
    "decode": {
      "prompt_tokens": 16,
      "decode_steps": 32,
      "reps": 5,
      "per_token_median_ms": 7.80,
      "tok_per_sec": 128.2
    }
  },
  "kernel_breakdown": {
    "f32_ms": 1.00,
    "f32_gbs": 28.5,
    "int8xf32_ms": 0.39,
    "int8xf32_gbs": 73.1,
    "int8xf32_xf32": 0.39
  }
}
```

### Batch Results (`batch.json`)

```json
{
  "$schema": "benchmark/batch-results.v1",
  "run_id": "anthony-laptop-smollm2-135m-q8-batch-a1b2c3d-20250106T120000Z",
  "engine": "fak multi-user batched decode Q8_0",
  "model": "SmolLM2-135M",
  "precision": "Q8_0",
  "workers": 16,
  "baseline": {
    "batch": 1,
    "tok_per_sec": 70.6
  },
  "peak": {
    "batch": 512,
    "agg_tok_per_sec": 862.2,
    "speedup_vs_baseline": 12.21,
    "speedup_vs_naive": 44.92
  },
  "points": [
    {
      "batch": 1,
      "per_user_ms_per_tok": 83.39,
      "agg_tok_per_sec": 11.99,
      "speedup_vs_baseline": 0.17
    },
    {"batch": 4, "per_user_ms_per_tok": 5.48, "agg_tok_per_sec": 182.43, "speedup_vs_baseline": 2.58},
    {"batch": 16, "per_user_ms_per_tok": 2.71, "agg_tok_per_sec": 368.75, "speedup_vs_baseline": 5.22},
    {"batch": 64, "per_user_ms_per_tok": 1.69, "agg_tok_per_sec": 590.72, "speedup_vs_baseline": 8.37},
    {"batch": 256, "per_user_ms_per_tok": 1.21, "agg_tok_per_sec": 825.57, "speedup_vs_baseline": 11.69},
    {"batch": 512, "per_user_ms_per_tok": 1.16, "agg_tok_per_sec": 862.19, "speedup_vs_baseline": 12.21}
  ]
}
```

### Master Catalog (`catalog.json`)

```json
{
  "$schema": "benchmark/catalog.v1",
  "version": "1.0",
  "last_updated": "2025-01-06T12:00:00Z",
  "machines": {
    "anthony-laptop": {
      "id": "anthony-laptop",
      "hostname": "ANTHONY-WIN",
      "os": "windows",
      "arch": "x86_64",
      "cpu_cores": 20,
      "gpu": "RTX 4070",
      "runs": 12,
      "last_run": "2025-01-06T12:00:00Z"
    },
    "mac-m3pro": {
      "id": "mac-m3pro",
      "hostname": "MacBook-Pro",
      "os": "macos",
      "arch": "arm64",
      "cpu_cores": 12,
      "gpu": "M3 Pro",
      "runs": 5,
      "last_run": "2025-01-05T10:30:00Z"
    }
  },
  "runs": [
    {
      "run_id": "anthony-laptop-smollm2-135m-q8-batch-a1b2c3d-20250106T120000Z",
      "machine_id": "anthony-laptop",
      "timestamp": "2025-01-06T12:00:00Z",
      "model": "SmolLM2-135M",
      "precision": "q8",
      "tags": ["kernel", "batch"],
      "peak_tok_per_sec": 862.2,
      "baseline_tok_per_sec": 70.6,
      "speedup": 12.21,
      "path": "runs/by-machine/anthony-laptop/20250106-120000-rtx4070-q8/"
    }
  ],
  "index": {
    "by_model": {
      "smollm2-135m": ["anthony-laptop-smollm2-135m-q8-batch-a1b2c3d-20250106T120000Z"]
    },
    "by_precision": {
      "q8": ["anthony-laptop-smollm2-135m-q8-batch-a1b2c3d-20250106T1200000Z"]
    },
    "by_date": {
      "2025-01-06": ["anthony-laptop-smollm2-135m-q8-batch-a1b2c3d-20250106T120000Z"]
    }
  }
}
```

## 3. Visualization Pipeline

### Chart Types

1. **Throughput vs Machine** (Bar chart)
   - X-axis: Machine names
   - Y-axis: Aggregate tokens/sec
   - Group by: Model, precision
   - Tool: Plotly interactive (default) or Matplotlib static

2. **Scaling Curves** (Line chart)
   - X-axis: Batch size
   - Y-axis: Aggregate throughput
   - Series: Different machines
   - Tool: Plotly interactive with hover details

3. **Cost-per-Token** (Table + optional chart)
   - Computed from hardware specs + throughput
   - Shows: $/1M tokens per machine

4. **Prefill vs Decode** (Grouped bar)
   - X-axis: Machine
   - Y-axis: tok/s (separate bars for prefill/decode)
   - Tool: Plotly

### Tooling

```bash
# Generate all charts for a model
python tools/bench_chart.py \
  --model smollm2-135m \
  --precision q8 \
  --output experiments/benchmark/charts/

# Generate specific comparison
python tools/bench_chart.py \
  --machines anthony-laptop,mac-m3pro \
  --run-type batch \
  --output experiments/benchmark/charts/batch-comparison.html

# Static PNG for docs
python tools/bench_chart.py \
  --model smollm2-135m \
  --format png \
  --output docs/benchmark/throughput-2025-01-06.png
```

### Chart Output

Interactive HTML (default):
- Embedded Plotly.js
- Hover tooltips with full details
- Filterable by machine/model/precision
- Export button (PNG/SVG)

Static outputs:
- `--format png`: High-res PNG for docs
- `--format svg`: Vector for publications
- `--format csv`: Raw data for external tools

## 4. Catalog/Index

### CLI Tool: `bench` (or `python tools/bench_cli.py`)

```bash
# List all runs
python tools/bench_cli.py list

# Filter by model
python tools/bench_cli.py list --model smollm2-135m

# Filter by machine
python tools/bench_cli.py list --machine anthony-laptop

# Filter by date range
python tools/bench_cli.py list --since 2025-01-01 --until 2025-01-07

# Show detailed run info
python tools/bench_cli.py show anthony-laptop-smollm2-135m-q8-batch-a1b2c3d-20250106T120000Z

# Compare runs
python tools/bench_cli.py compare \
  --run-id anthony-laptop-smollm2-135m-q8-batch-a1b2c3d-20250106T120000Z \
  --run-id mac-m3pro-smollm2-135m-q8-batch-a1b2c3d-20250106T120000Z

# Generate comparison table
python tools/bench_cli.py table \
  --model smollm2-135m \
  --precision q8 \
  --format markdown
```

### Query Examples

```bash
# Find fastest run for a model
python tools/bench_cli.py best --model smollm2-135m --metric peak_tok_per_sec

# Find all runs on a specific machine
python tools/bench_cli.py list --machine anthony-laptop --format json

# Summary stats by machine
python tools/bench_cli.py summary --group-by machine

# Detect regressions (compare latest vs baseline)
python tools/bench_cli.py regression \
  --machine anthony-laptop \
  --baseline run-id-12345
```

### Web Dashboard (Optional, Nice-to-Have)

Simple static HTML dashboard generated by CLI:

```bash
python tools/bench_cli.py dashboard --output experiments/benchmark/dashboard/index.html
```

Features:
- Machine inventory table
- Recent runs list
- Top charts
- Search/filter
- No backend required (pure static)

## 5. Onboarding

### One-Time Machine Setup Script

```bash
# On the new machine, from the repo root:
python tools/bench_onboard.py --interactive

# Or non-interactive:
python tools/bench_onboard.py \
  --machine-id new-node-01 \
  --tags "gpu,a100,linux"
```

The script:
1. Detects hardware (CPU, GPU, RAM, storage)
2. Detects OS/runtime versions
3. Generates `specs.json`
4. Runs a smoke benchmark
5. Validates the harness works
6. Updates the catalog

### Machine Specs Template

Generated automatically, but template for reference:

```json
{
  "$schema": "benchmark/machine-specs.v1",
  "machine_id": "{{MACHINE_ID}}",
  "hostname": "{{HOSTNAME}}",
  "registered_at": "{{NOW}}",
  "hardware": {
    "cpu": {
      "model": "{{CPU_MODEL}}",
      "cores_physical": {{PHYSICAL_CORES}},
      "cores_logical": {{LOGICAL_CORES}}
    },
    "gpu": [
      {
        "model": "{{GPU_MODEL}}",
        "memory_gb": {{GPU_MEM}},
        "compute_capability": "{{COMPUTE_CAP}}"
      }
    ],
    "ram_gb": {{RAM_GB}}
  },
  "os": {
    "name": "{{OS_NAME}}",
    "version": "{{OS_VERSION}}"
  },
  "runtime": {
    "go_version": "{{GO_VERSION}}",
    "python_version": "{{PY_VERSION}}"
  },
  "tags": [{{ARCH}}, {{HAS_GPU}}, {{OS_FAMILY}}]
}
```

### First Benchmark Run Template

After onboarding, the template first run:

```bash
# Generated by onboard script, or run manually:
python tools/bench_run.py \
  --machine-id $(cat .bench-machine-id) \
  --model smollm2-135m \
  --precision q8 \
  --config batch
```

### Validation Checklist

Before a machine is considered "production" for benchmarking:

- [ ] Specs file exists and is valid JSON
- [ ] All required fields populated (CPU, GPU, RAM, OS)
- [ ] Smoke benchmark completes without error
- [ ] At least one full benchmark run with gate pass
- [ ] Catalog includes the machine
- [ ] At least one comparison exists (vs another machine or baseline)

## Migration Plan

### Phase 1: Infrastructure Setup (Week 1)

1. Create directory structure under `experiments/benchmark/`
2. Implement schema validators (JSON Schema)
3. Build catalog indexer
4. Implement CLI `list` and `show` commands

### Phase 2: Machine Migration (Week 2)

1. Run `bench_onboard.py` on existing nodes
2. Convert existing `fleet-nodes/` results to new format
3. Import historical runs into catalog
4. Validate conversion with comparison tables

### Phase 3: Visualization (Week 3)

1. Implement chart generator (Plotly)
2. Add `bench chart` CLI command
3. Generate dashboard HTML
4. Add to existing docs

### Phase 4: Automation (Week 4)

1. Add to existing harness scripts
2. Auto-update catalog on run completion
3. Add regression detection
4. Integrate with CI (optional)

### Backward Compatibility

Existing paths remain valid:
- `fak/experiments/fleet-nodes/` → Migrate to `experiments/benchmark/runs/by-machine/`
<!-- `fak/experiments/dgx/` excluded from the public copy (operator-private lab infra). -->
- `fak/experiments/model-baseline/` → Cross-reference from catalog

## Implementation Priority

1. **High Priority**
   - Schema definitions (JSON Schema files)
   - Catalog indexer (`tools/bench_catalog.py`)
   - CLI list/show (`tools/bench_cli.py`)
   - Migration script for existing data

2. **Medium Priority**
   - Visualization (`tools/bench_chart.py`)
   - Onboarding script (`tools/bench_onboard.py`)
   - Regression detection

3. **Low Priority**
   - Web dashboard
   - Cost-per-token calculations
   - Advanced queries (aggregations, trends)

## File Paths Reference

All paths relative to repo root:

```
experiments/benchmark/
  ├── catalog.json
  ├── machines/
  │   ├── catalog.json
  │   └── <machine-id>/
  │       └── specs.json
  ├── runs/
  │   ├── by-machine/<machine-id>/<timestamp>-<run-hash>/
  │   └── by-model/<model-id>/<run-id>.json (symlinks or copies)
  └── charts/
      ├── <chart-name>.html
      └── <chart-name>.png

tools/
  ├── bench_catalog.py        # Catalog maintenance
  ├── bench_cli.py            # Main CLI (list, show, compare, table)
  ├── bench_chart.py          # Visualization generator
  ├── bench_onboard.py        # New machine setup
  ├── bench_migrate.py        # Migrate existing data
  └── schemas/
      ├── machine-specs.v1.json
      ├── run-manifest.v1.json
      ├── kernel-results.v1.json
      ├── batch-results.v1.json
      └── catalog.v1.json
```

## Extension Points

The design supports future additions:
- New benchmark types (add to `manifest.artifacts`)
- New metrics (extend result schemas)
- New chart types (add to `bench_chart.py`)
- New query patterns (extend CLI)

All changes are additive and backward compatible.
