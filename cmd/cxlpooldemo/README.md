# CXL Pool Demo

`cxlpooldemo` is a no-model, no-GPU cost-model demo for a coherent, switch-pooled memory tier. It shows when one resident KV prefix can be shared across a fleet, and when the trust gate refuses cross-tenant reuse.

## Prerequisites

Requires Go only. The default path uses representative profiles embedded in the code. An optional JSON calibration file can override the profile numbers.

## Quick Start

From the repo root:

```bash
go run ./cmd/cxlpooldemo
go run ./cmd/cxlpooldemo -profiles cmd/cxlpooldemo/calibration.example.json
```

Each run completes in a few seconds and returns exit code 0 on success. The default output is deterministic; the calibrated output is deterministic for the supplied JSON file.

## What You See

```text
Profiles: representative default profiles
== Pool topology (who can attend each tier) ==
== 8 tenants share one hot 4000-token system+tool prefix (64MB KV) ==
  -> a coherent CXL pool turns 8 prefills into 1
== Cross-tenant reuse gate (dedup is honest, not blind aliasing) ==
```

## What This Does Not Claim

This demo does not claim to measure real CXL hardware, move tensors, or allocate a physical shared region. It computes fleet economics over the tier and pool profiles you give it.

## Related Docs

- [CXL memory pool](../../docs/serving/cxl-memory-pool.md)
- [Hardware-aware cache](../../docs/serving/hardware-aware-cache.md)
