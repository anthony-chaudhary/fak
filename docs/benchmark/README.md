# Benchmark Documentation

This directory contains comprehensive documentation for the fleet benchmark infrastructure.

> **📊 Authority:** All committed benchmark performance claims are centrally indexed in
> **[fak/BENCHMARK-AUTHORITY.md](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)** with full traceability
> to commits and artifacts. See **[fak/BENCHMARK-GOVERNANCE.md](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-GOVERNANCE.md)**
> for the DOS-centric process that creates, verifies, and publishes benchmark results.

## Quick Links

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](QUICKSTART.md) | Get started with cross-machine benchmarks |
| [CROSS-MACHINE-INFRASTRUCTURE.md](CROSS-MACHINE-INFRASTRUCTURE.md) | Full design spec and schema reference |

## Infrastructure Overview

The benchmark infrastructure enables:
- **Multi-machine tracking**: Compare results across Mac, Windows, Linux
- **Queryable catalog**: Find runs by machine, model, precision, date
- **Visualization**: Interactive charts for throughput, scaling, cost
- **Onboarding**: Simple setup for new benchmark nodes

## Core Tools

| Tool | Purpose |
|------|---------|
| `tools/bench_catalog.py` | Build/update master catalog |
| `fak bench-runs` | Query and compare results |
| `tools/bench_chart.py` | Generate visualizations |
| `tools/bench_onboard.py` | Register a new machine |
| `tools/bench_migrate.py` | Migrate existing data |

## Key Concepts

### Run Identifiers

Format: `<machine-id>-<model-id>-<precision>-<config-hash>-<timestamp>`

Example: `anthony-laptop-smollm2-135m-q8-batch-a1b2c3d-20250106T120000Z`

### Storage Structure

```
experiments/benchmark/
├── catalog.json              # Master index
├── machines/                 # Machine registry
├── runs/                     # All benchmark results
└── charts/                   # Generated visualizations
```

### Schema Validation

All artifacts use JSON Schema validation:
- `tools/schemas/machine-specs.v1.json` - Machine specifications
- `tools/schemas/run-manifest.v1.json` - Run metadata
- `tools/schemas/kernel-results.v1.json` - Kernel benchmark results
- `tools/schemas/batch-results.v1.json` - Batched decode results
- `tools/schemas/catalog.v1.json` - Master catalog

## Quick Reference

### First-time Setup

```bash
# 1. Onboard machine
python tools/bench_onboard.py --interactive

# 2. Migrate existing data (optional)
python tools/bench_migrate.py --apply

# 3. Build catalog
python tools/bench_catalog.py build
```

### Daily Operations

```bash
# Update catalog after new run
python tools/bench_catalog.py update

# List runs
fak bench-runs list

# Generate charts
python tools/bench_chart.py all
```

### Onboarding New Machine

```bash
# On the new machine
python tools/bench_onboard.py --interactive

# Then on any machine (update catalog)
python tools/bench_catalog.py update

# Verify
fak bench-runs list --machine <new-machine-id>
```

## Related Methodology Documents

- [Production Benchmark Methodology](../production-benchmark-methodology.md) - Phase 0 kernel benchmarking
<!-- GPU server Benchmark Methodology excluded from the public copy (operator-private
     lab infra). See PUBLIC-SCRUB-POLICY.md PRIVATE-ONLY list. -->
- [Permission System Benchmark](../permission-system-benchmark-methodology.md) - Security boundary comparison
