---
title: "Micro-context S5: controlled CUDA-kernel 10k soak"
description: "Status: PASS on August 9, 2026. The captured ledger s5-gcp-10k-controlled-soak-2026-08-09.json closes the remaining S5 controlled-kernel evidence gap in 5792."
---
# Micro-context S5: controlled CUDA-kernel 10k soak

**Status:** PASS on August 9, 2026. The captured ledger
[`s5-gcp-10k-controlled-soak-2026-08-09.json`](../../experiments/microcontext/s5-gcp-10k-controlled-soak-2026-08-09.json)
closes the remaining S5 controlled-kernel evidence gap in #5792.

The sanctioned L4 node exposes three distinct paths. Keeping them distinct matters:

- native CPU `fak serve` (`qwen2.5-0.5b-cpu`);
- fak proxy to Ollama/L4 (`qwen2.5:14b`);
- native controlled CUDA `fak serve` (`qwen2.5-0.5b-gpu`).

The final S5 run used the native controlled-CUDA path. Public evidence scrubs the
endpoint address.

## Final 10,000-context witness

| Dimension | Observed result |
|---|---:|
| Logical contexts / physical workers | 10,000 / 2 |
| Exact retirement accounting | 10,000 completed, 0 failed |
| Usage-bearing model turns | 10,000 |
| Wall time | 14,165.405 s (3.9348 h) |
| TTFT p50 / p95 / max | 2.832 / 2.873 / 2.926 s |
| Completion latency p50 / p95 / max | 2.832 / 2.873 / 2.926 s |
| Prompt / completion tokens | 7,048,890 / 80,000 |
| Aggregate prompt / decode rate | 497.61 / 5.65 tok/s |
| Verified nonempty results | 0.706 per wall second |
| Independent resource samples | 472, with 0 sample errors |
| Client peak RSS | 31,883,264 bytes (30.41 MiB) |
| Server peak RSS / Go heap | 5,305,233,408 / 4,631,053,864 bytes |
| Sampled device memory / power peaks | 2,312 MiB / 35.86 W |

The interactive dimension is the client-observed TTFT/tail distribution. The
aggregate dimension is usage-bearing tokens and verified nonempty results divided
by total wall time. Neither is substituted for the other.

## Controlled failure and residency contract

The `-controlled-soak` path executes the named transitions rather than stamping
declared counters:

1. A real endpoint canary runs under a candidate shared base.
2. The endpoint rolls back to the canonical base, and canary telemetry is reset so
   it cannot inflate the measured 10,000-turn workload.
3. The 9,998 contexts outside the two-worker resident set are frozen through
   `microagent.HibernationStore` and restored byte-identically.
4. One controlled overload and one controlled cancellation are injected at the
   gateway seam and recovered.
5. Four additional provider transient failures occurred during the live workload;
   all four recovered within the hard three-attempt ceiling.

The independent endpoint counter increased by exactly 10,001: one canary plus
10,000 measured model turns. The server start witness did not change during the
run, so this pass did not cross a server restart. The ledger embeds the complete
30-second resource trace, the run-binary/raw-report/trace SHA-256 digests, empty
stderr evidence, and exit code 0.

## Scale ladder and admission findings

| Path | Contexts/workers | Wall | TTFT p50/p95 | Prompt/completion | Aggregate prompt/decode |
|---|---:|---:|---:|---:|---:|
| Native CUDA, Qwen 0.5B Q8 | 100 / 1 | 113.840 s | 1.142 / 1.230 s | 70,290 / 800 | 617.44 / 7.03 tok/s |
| Native CUDA, Qwen 0.5B Q8 | 1,000 / 1 | 1,045.554 s | 1.058 / 1.144 s | 703,890 / 8,000 | 673.22 / 7.65 tok/s |
| Native CUDA, Qwen 0.5B Q8 | 10,000 / 2 | 14,165.405 s | 2.832 / 2.873 s | 7,048,890 / 80,000 | 497.61 / 5.65 tok/s |
| Ollama L4 proxy, Qwen 14B | 100 / 1 | 398.160 s | 3.943 / 4.201 s | 70,290 / 794 | 176.54 / 1.99 tok/s |

A fresh 20-context four-worker probe failed after recovering 13 of 17 provider
transients; two streams still lacked complete usage. A two-worker probe passed
20/20 and therefore selected the final run envelope. This reinforces the earlier
finding that software concurrency is not free physical concurrency: admission must
follow the endpoint's observed stable envelope.

The native-CUDA and Ollama rows are **not a same-model backend A/B** because model
sizes differ. They prove those specific paths and workloads, not a general backend
speedup.

## Reproduce and verify

Synthetic regression floor:

```powershell
go run ./cmd/microcontextdemo -selfcheck -contexts 10000 -workers 64
```

Controlled live run, with the sanctioned endpoint supplied by the operator:

```powershell
go run ./cmd/microcontextdemo `
  -contexts 10000 -workers 2 `
  -endpoint $ENDPOINT `
  -model qwen2.5-0.5b-gpu `
  -provider fak-native-cuda-openai-compatible `
  -hardware GCP-sanctioned-L4-controlled-CUDA `
  -controlled-soak -request-timeout 5m -run-timeout 0
```

Artifact verification:

```powershell
go run ./cmd/microcontextdemo -verify experiments/microcontext/s5-gcp-10k-controlled-soak-2026-08-09.json
```

The earlier scale artifacts remain independently verifiable:

```powershell
go run ./cmd/microcontextdemo -verify experiments/microcontext/s5-gcp-100-cuda-workers1-pass-2026-08-06.json
go run ./cmd/microcontextdemo -verify experiments/microcontext/s5-gcp-100-l4-pass-2026-08-06.json
go run ./cmd/microcontextdemo -verify experiments/microcontext/s5-gcp-1000-cuda-workers1-pass-2026-08-06.json
```

## Claim boundary

- Rates are aggregate observed provider usage divided by wall time, not
  server-internal kernel rates.
- Completion checks prove transport, nonempty output, exact usage, and exact
  retirement, not semantic task quality.
- The endpoint exposed no resident-KV capacity counter, so the ledger makes no
  KV-capacity claim.
- Per-item queue age was not exposed, so the ledger records the bounded queue,
  resident/hibernated counts, and endpoint in-flight peak but makes no queue-age
  percentile claim.
- Thirty-second point sampling observed device memory and power but did not
  intersect an active kernel, so no GPU-utilization claim is made.


