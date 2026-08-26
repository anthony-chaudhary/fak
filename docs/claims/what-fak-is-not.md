---
title: "What fak is not: explicit product boundaries"
description: "This boundary ledger names unbuilt fak capabilities, including external-engine KV attachment and a fine-tuned syscall model, without overstating shipped seams."
---

# What fak is NOT

[← Claims index](../../CLAIMS.md)


- [STUB] No LIVE transport attaching a real *external* serving engine's KV region — vLLM/SGLang owns the KV in Python/CUDA, so importing its pinned pages into an `Arena` over CUDA-IPC / shared memory is the engine-specific transport still to build (`xenginekv.AttachArena` is the buffer entry point it plugs into; in-process the arena is a Go `[]byte` stand-in, not a mapped engine region). The cross-engine zero-copy KV co-residence SEAM itself shipped in #448 (`internal/xenginekv`, the SHIPPED row above) — the frozen `RegisterRegionBackend`/`RegisterPageOutBackend` ABI, the zero-copy Resolve view, and the region-addressed `Evict`/`Clone` quarantine that makes the per-agent KV-fusion hold against an engine fak does not itself run — so this remaining transport is a backend plug-in behind that frozen ABI, no further ABI change. (v0.2's in-kernel model owns *its own* KV cache — that is the original fusion; the seam above is its cross-engine dual.)
- [STUB] No FINE-TUNED *syscall/adjudication* LLM and no AsyncLM interrupt behavior — the harvest-corpus consumer edge CLOSED (#580) by a SMALL classifier (`internal/advmodel`, the SHIPPED row above): it trains on the floor-labeled corpus and emits a fail-closed advisory signal, but it is a logistic-regression bag-of-tokens model, NOT a fine-tune of the fused SmolLM2 forward pass. The model fused in v0.2 remains a *stock* SmolLM2 reference forward pass; training/grafting a tuned adjudication head onto that fused model (GPU + base weights + multi-hour training) is still unbuilt, as is AsyncLM's interrupt behavior.
- [SIMULATED] token-per-watt is read-only SIMULATED telemetry because there is no watt source on the box. Native continuous batching is no longer in this bucket for the in-kernel lifecycle path (#401), but production-grade multi-tenant p99 scheduling is still a separate honest no-claim. NOTE: "no GPU dependency" is no longer strictly true — the optional `-tags vulkan` AMD backend runs the model on a real RX 7600 — but it is OFF by default; the shipped pure-Go binary still has zero GPU dependency.
