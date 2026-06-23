# Cross-Machine Benchmark Quick Start

**Last Updated:** 2025-01-06 | **Status:** v1.0

This guide gets you started with the cross-machine benchmark infrastructure.

## Initial Setup (One Time)

### 1. Onboard Your First Machine

```bash
# Interactive mode (recommended for first machine)
python tools/bench_onboard.py --interactive

# Or non-interactive with explicit values
python tools/bench_onboard.py \
  --machine-id anthony-laptop \
  --tags "gpu,rtx4070,windows,wsl2"
```

This creates `experiments/benchmark/machines/<machine-id>/specs.json`.

### 2. Migrate Existing Data (Optional)

If you have existing benchmark runs:

```bash
# Preview what will be migrated
python tools/bench_migrate.py --dry-run

# Apply migration
python tools/bench_migrate.py --apply
```

### 3. Build the Initial Catalog

```bash
python tools/bench_catalog.py build
```

This creates `experiments/benchmark/catalog.json` from all discovered runs.

## Daily Workflow

### Add a New Benchmark Run

After running a benchmark (using existing harness), the results go into:
```
fak/experiments/benchmark/runs/by-machine/<machine-id>/<timestamp>-<config>/
```

Then update the catalog:

```bash
python tools/bench_catalog.py update
```

### Query Results

```bash
# List all runs
python tools/bench_cli.py list

# Filter by machine
python tools/bench_cli.py list --machine anthony-laptop

# Filter by model
python tools/bench_cli.py list --model smollm2-135m

# Show detailed run info
python tools/bench_cli.py show <run-id>

# Compare two runs
python tools/bench_cli.py compare <run-id-1> <run-id-2>

# Find the best run for a model
python tools/bench_cli.py best --model smollm2-135m

# Generate comparison table (Markdown)
python tools/bench_cli.py table --model smollm2-135m
```

### Generate Charts

```bash
# Generate all charts
python tools/bench_chart.py all --output-dir experiments/benchmark/charts/

# Specific chart types
python tools/bench_chart.py throughput --model smollm2-135m
python tools/bench_chart.py scaling --machines anthony-laptop,mac-m3pro
```

Charts are saved as interactive HTML files with Plotly.

## Onboarding a New Machine

### Step 1: Run Onboarding Script

On the new machine:

```bash
python tools/bench_onboard.py --interactive
```

This detects CPU, GPU, RAM, OS, and runtime versions.

### Step 2: Run a Smoke Benchmark

```bash
bash tools/fak_node_bench.sh --short --host=<machine-id>
```

### Step 3: Update Catalog

```bash
python tools/bench_catalog.py update
```

### Step 4: Verify

```bash
python tools/bench_cli.py list --machine <machine-id>
```

## Directory Structure Reference

```
experiments/benchmark/
├── catalog.json              # Master index (run after each benchmark)
├── machines/                 # Machine registry
│   └── <machine-id>/
│       └── specs.json       # One-time hardware specs
├── runs/                    # All benchmark results
│   └── by-machine/
│       └── <machine-id>/
│           └── <timestamp>-<config>/
│               ├── manifest.json     # Run metadata
│               ├── kernel.json       # Kernel results
│               ├── batch.json        # Batch results
│               └── fleetbench.json   # Fleet turn-tax
└── charts/                   # Generated visualizations
    ├── throughput.html
    ├── scaling.html
    └── prefill-decode.html
```

## Validation

### Validate Catalog Integrity

```bash
python tools/bench_catalog.py validate
```

### Check a Specific Run

```bash
python tools/bench_cli.py show <run-id>
```

## Common Workflows

### Compare Performance Across Machines

```bash
# Get comparison table
python tools/bench_cli.py table --model smollm2-135m --format markdown

# Generate scaling curve chart
python tools/bench_chart.py scaling --model smollm2-135m
```

### Track Performance Over Time

```bash
# List all runs for a machine, sorted by date
python tools/bench_cli.py list --machine anthony-laptop

# Compare latest vs previous
python tools/bench_cli.py compare <latest-id> <previous-id>
```

### Find Performance Regression

```bash
# List all runs for a model
python tools/bench_cli.py list --model smollm2-135m

# Check if latest is slower than best
python tools/bench_cli.py best --model smollm2-135m
python tools/bench_cli.py compare <latest> <best-id>
```

## Troubleshooting

### Catalog is empty or outdated

```bash
# Rebuild from scratch
python tools/bench_catalog.py build
```

### Machine not showing in catalog

```bash
# Ensure specs.json exists
ls experiments/benchmark/machines/<machine-id>/specs.json

# Rebuild catalog
python tools/bench_catalog.py build
```

### Charts not generating

Ensure catalog is built first:

```bash
python tools/bench_catalog.py build
python tools/bench_chart.py all
```

## Next Steps

- Add machines to the catalog as they come online
- Run benchmarks regularly (daily/weekly) and update catalog
- Use charts to track performance trends
- Use `compare` to validate tuning changes

## Related Documentation

- [Full Infrastructure Design](CROSS-MACHINE-INFRASTRUCTURE.md)
- [Phase 0 Methodology](../production-benchmark-methodology.md)
<!-- GPU server Methodology excluded from the public copy (operator-private lab infra). -->
