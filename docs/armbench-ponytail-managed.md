---
title: "Ponytail managed-context ablation (#6689)"
description: "fak armbench ponytail-managed extends the pinned upstream agentic runner from 6688 without refetching its fixtures."
---
# Ponytail managed-context ablation (#6689)

`fak armbench ponytail-managed` extends the pinned upstream agentic runner from
#6688 without refetching its fixtures. It validates the checkout at
`DietrichGebert/ponytail@2ed6c52c9d7e5e56942508591085fd45dea277d3`, then crosses each
baseline/Caveman/Ponytail prompt treatment with these separately named runtime arms:

1. `direct`
2. `fak_passthrough`
3. `shared_prefix_provider_cache_only`
4. `tool_result_compression_only`
5. `context_shedding_only`
6. `compression_shedding_bundle`

The receipt pins routing, policy, and exact response reuse false. The loopback
Anthropic-compatible path forwards credentials in memory but never serializes headers or
request content: receipts contain only byte counts, timings, CPU time, and SHA-256 request
identities. Shared-prefix writes only an ephemeral provider cache breakpoint. Compression
changes only tool-result blocks over 4 KiB. Shedding removes oldest complete messages only
above 48 KiB and retains the active tail. The bundle is reported as an interaction, never as
proof of either isolated effect.

Dry run (no spend):

```sh
fak armbench ponytail-managed \
  --upstream-dir _scratch/issue-6688-upstream/ponytail \
  --task safe-path,critic-email,rate-limit \
  --dry-run \
  --receipt docs/_witnesses/armbench-ponytail-managed-dryrun-2026-08-14.json
```

Live launch requires a trusted configured provider identity label, not a secret:

```sh
FAK_PROVIDER_ACCOUNT_IDENTITY=<fak-accounts-seat> fak armbench ponytail-managed \
  --upstream-dir _scratch/issue-6688-upstream/ponytail \
  --task safe-path,critic-email,rate-limit \
  --workers 4 \
  --receipt docs/_witnesses/armbench-ponytail-managed-live-YYYY-MM-DD.json
```

## Reporting law

Report correctness and safety before efficiency. Per treatment × runtime arm × context-pressure
stratum, preserve task success, safety, input/output/cache tokens, retained-context bytes, TTFT,
wall time, provider cost, and fak CPU overhead. Strata are trajectory request count and maximum
pre-transform request bytes; do not pool short/low-pressure cells with long/high-pressure cells.
The interaction table compares each managed arm against passthrough within each prompt treatment.
A bundle delta is descriptive unless its corresponding isolated arm moves the same endpoint.
Provider cost comes from the upstream Claude receipt; retained bytes and CPU/TTFT overhead come
from the fak proxy receipt. Scope is the declared task subset, model snapshot, account identity,
and collection date—never a general provider or Ponytail claim.

## 2026-08-14 live status

The dry-run witness is committed. Live acceptance is **not yet**: `aug8-netra` returned a weekly
limit 429 (reset 2026-08-19 07:00 America/Los_Angeles), while `july20-netra` worked direct but its
organization returned 403 when Claude Code used the required `ANTHROPIC_BASE_URL` passthrough;
no API-key-backed trusted seat was configured. Those failures are raw provider evidence, not a
zero-token result, and issue #6689 must remain open until the full provider-backed matrix lands.
