# Retiring the vendored Inkling Metal runtime

**Audience:** maintainers deciding what runs Inkling-2-Mini on an Apple GPU. The native path is described in [`inkling-metal-runtime.md`](inkling-metal-runtime.md); this page records why the vendored runtime left public fak and what it is still measured against.

Status: removal lands in the fak run and fak serve lanes; until then trunk may still accept the vendored selector.

## Decision

Public fak ships one Inkling-2 Metal path: fak's own resident device (`compute.Inkling2Device` over `internal/metalgemm`). The vendored Splash runtime (`incoai/splash`, Apache-2.0) is no longer built, linked or selectable here. It moves to fak-private as a comparison bar, built there as `cmd/fak-splash-serve`, an OpenAI-compatible server that the native `fak serve` is benchmarked against.

Labels stay honest in both directions: no public path reports native work under a vendored name, and the comparison build never labels itself fak, native or resident.

## What was retired

- `FAK_INKLING_METAL_TARGET=vendored` is refused with a message pointing to the fak-private comparison build.
- With the variable unset, the target is `resident` (the host-driven `native` forward remains as the fallback where the resident device is not linked).
- The vendored trees, their Go binding and the provenance pin leave this repo. The Splash entry in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) stays, because fak's own `ink2_*.metal` kernels and `internal/kvmmu` still adapt Splash techniques.

## Comparison receipts

All rows use `Inkling-2-Mini-UD-Q3_K_XL.gguf`, greedy (`--temp 0`), on the M3 Pro 36 GiB reference machine. *Measured (contended)* means other sessions loaded the machine: ratios between paths are meaningful, absolute numbers are not release figures. *Author-measured* means no independent reviewer reran it.

Last recorded vendored numbers (the scoreboard at retirement):

| Vendored row | Value | Evidence |
|---|---|---|
| Decode, DFlash2 draft on, 31-token prompt | 42.5 tok/s (38.9–44.6) | *Measured (contended)*, load average 16–191 |
| Decode, draft off | 18.4 tok/s | *Measured (contended)*, same session |
| Prefill | about 425 tok/s (165–497 across prompt lengths) | *Measured* baseline on fak 0.57.0 |
| Aggregate decode, no draft | 40.6 tok/s at 2 exact lanes; 58.2 at 4 wide lanes (not bit-reproducible) | *Measured (contended)* or *author-measured* |
| Prefix reuse, 4250-token prompt | time to first token 8.95 s -> 0.50 s | *Measured (contended)* or *author-measured* |
| Long-context decode | 37.8 tok/s at 8k; 35.9 at 32k | *Measured (contended)* or *author-measured* |

Recorded serve h2h, 2026-11-26 (`fak bench h2h`, M3 Pro, cold 512-token prompt, 128 decode tokens, speculation off):

| Concurrent requests | Resident total decode | Vendored total decode |
|---|---|---|
| 1 | 27.7 tok/s | 25.0 tok/s |
| 2 | 26.4 tok/s | 20.4 tok/s |
| 4 | 21.9 tok/s | 19.4 tok/s |

Cold prefill under serve: resident 224 tok/s, vendored 465 tok/s.

## The tradeoff

The default flip skipped the stage-1 parity gate this page used to set (native at or above vendored on decode and prefill before it becomes the default). Resident leads on non-speculative serve decode, but it still trails vendored on prefill (about 224 vs 465 tok/s under serve) and on speculative decode (vendored DFlash2 at 42.5 tok/s). Native clarity was chosen over the gate; those two gaps are the open work, and the fak-private comparison build is how they are measured.
