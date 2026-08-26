---
title: "fak tool vDSO: three-tier local fast path"
description: "fak's tool vDSO serves eligible read-only idempotent calls through a pure registry, versioned content cache, or static table with canonical JSON keys."
---

# Tool vDSO (3-tier local fast path)

[← Claims index](../../CLAIMS.md)


- [SHIPPED] Tier-1 pure registry (gated on readOnlyHint+idempotentHint, re-checked not trusted), tier-2 content-addressed cache (world-versioned, LRU), tier-3 static table. Witness: `vdso` tests (units 25–38). Prior art: kernel vDSO; RadixAttention prefix reuse.
- [SHIPPED] Arg-order-independent content keys (canonicalized JSON). Witness: vdso canonicalization test (unit 26).
- [SHIPPED] A write-shaped completion bumps the world-version and invalidates the cache (soundness: a hit equals a fresh call). Witness: units 28, 38.
- [SIMULATED] Real-world vDSO hit-rate: the demo trace `tau2-smoke` is deliberately cache-favorable (~50% hits). The EXPERIMENTS measured addressable purity on real tau2-airline is ~0.7% — far below a useful threshold. The vDSO is therefore an UPSIDE secondary, never the headline. Witness: `report.json` `vdso_hit_rate` (reported, never gated, unit 33/83).
