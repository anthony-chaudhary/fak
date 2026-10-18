---
title: "QEvict recoverable quantized KV eviction evaluation"
description: "Status: research record of a retired bounded contract for issue 6258. This is not a model-quality or production-throughput claim."
---
# QEvict recoverable quantized KV eviction evaluation

**Status:** research record of a retired bounded contract for issue #6258. The
`internal/qevicteval` package it described had no caller in fak and was removed; it remains in
git history. This is not a model-quality or production-throughput claim.

## Named research object

The retired leaf evaluated **QEvict: Recoverable Quantized KV Eviction for
Attention-Drift-Robust Long-Context Decoding**, Garg et al.,
[arXiv:2608.05326v1](https://arxiv.org/abs/2608.05326v1), published 2026-08-05.
The pinned paper PDF used for review was 16,507,328 bytes with SHA-256
`baad72aecfec9adf0301aafb82c6a81ecdb4fb90b42a1ea2b15c5cdcc1f9b5dd`.
The recipe identity in the contract is `qevict/arxiv-2608.05326v1`; no generic
"recoverable cache" surrogate is substituted.

QEvict's named mechanism is a three-tier cache: important windows remain full precision,
intermediate windows remain recoverable in quantized form, and the lowest-confidence
windows are deleted. A later-important recoverable window is read, dequantized, and
promoted. The paper names Future Missed Mass and Global LIR as drift diagnostics. The
leaf measured the directly trace-replayable subset: future attention mass lost by ordinary
irreversible eviction versus deletion by QEvict, recovery events/read bytes, and capacity.
It did not pretend that the bounded trace reproduces Global LIR or downstream benchmark
quality.

## Contract

The retired `internal/qevicteval` accepted only `qevicteval/v1`, the exact recipe pin above, a
content-addressed trace, and `qevicteval/trace-replay-v1`. Results were typed:

- `supported`: pinned recipe and runtime were evaluated; decision is `integrate` or
  `abstain` from the bounded evidence.
- `unsupported`: unknown contract/recipe, invalid artifact provenance, or invalid trace.
- `delegate`: the runtime is unknown or an independently captured runtime observation is
  absent. There is no silent fallback to modeled latency.

The result kept capacity/drift evidence and latency evidence separate. Trace-derived
bytes and missed mass are **modeled**. Latencies are **observed** only when platform,
device, command, capture time, and evidence digest are pinned.

## Reproducible witness

The three independent JSON fixtures in the retired package's testdata covered recovery,
no recovery benefit, and runtime delegation. The recovery fixture contained a historical
window ordinary eviction deletes but QEvict preserves at 25 bytes and later recovers.
Its deterministic replay reported:

| Metric | Ordinary eviction | QEvict | Evidence |
|---|---:|---:|---|
| Peak bytes in fixture accounting | 100 | 150 | modeled from fixture |
| Future missed attention mass | 0.41 | 0.01 | modeled from fixture |
| Recovery events / bytes read | 0 / 0 | 1 / 25 | modeled from fixture |
| Replay loop median (5 runs, 100,000 iterations) | 3.302 ns/op | 4.349 ns/op | observed, WSL Linux amd64, AMD Ryzen 9 9950X CPU |
| Recovery-copy median | n/a | 0.6407 ns/op | observed, same envelope |

The captured benchmark output (SHA-256 `f7506bea881e438501367ca73f1fbce8d8dc83f9f9832e4e8527068d1a480779`) was pinned by the observed fixtures and remains in git history. These nanobenchmarks measured
the Go trace adjudicator and a 25-byte recovery copy, **not** attention kernels,
dequantization, GPU memory traffic, token throughput, or model quality. Therefore the
observed runtime envelope supported the contract path only. Production integration remains
an abstain/delegate until a pinned implementation artifact and model/hardware witness exist.
No GPU evidence is required to establish this deliberately CPU-only contract witness; any
future CUDA claim must use the sanctioned lab path in `docs/private-comms-channel.md` and
must replace neither these provenance pins nor the modeled/observed distinction.

## Decision

`integrate` meant only that this exact bounded trace contained a recoverable future-attention
benefit and a complete runtime observation. `abstain` was returned when no recovery benefit
was present. Unknown or unobserved environments delegated. No universal quantization winner,
quality gain, or production performance gain is claimed.
