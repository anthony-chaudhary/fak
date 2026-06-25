# Hardware-Aware Cache Demo

`hwcachedemo` is a no-model, no-GPU proof that fak's cache placement policy uses hardware cost rather than blind LRU. It prints the residency ladder and shows a hot prefix demoted to colder memory instead of evicted and re-prefilled.

## Prerequisites

Requires Go only. It does not need a model, GPU, CXL device, network, or local calibration file.

## Quick Start

From the repo root:

```bash
go run ./cmd/hwcachedemo
```

The run completes in a few seconds and returns exit code 0 on success. Output is deterministic because the demo injects a fixed logical clock and fixed tier profiles.

## What You See

```text
== Residency tier ladder (hot -> cold) ==
== A hot 4000-token prefix under escalating memory pressure ==
  step 0: demote  hbm -> dram
== Blind LRU vs hardware-aware tiering (8 turns sharing a 4000-token prefix) ==
  -> 28000 prefill tokens saved by demoting instead of evicting
```

## What This Does Not Claim

This demo does not claim live serving integration or physical KV movement. It exercises the metadata and policy plane that decides when demote beats evict.

## Related Docs

- [Hardware-aware cache](../../docs/serving/hardware-aware-cache.md)
- [Hardware limits and capacity](../../docs/explainers/hardware-limits-and-capacity.md)
