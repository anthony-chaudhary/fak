---
title: "Campaign record: Native two-card fit candidate — 2026-10-26"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak anthony-27b-multigpu-fit (2026-10-26): Native two-card fit candidate — 2026-10-26."
---

# Native two-card fit candidate — 2026-10-26

Status: **software candidate committed; CUDA build and physical 27B run blocked**. No new GPU fit, prefill/decode rate, memory or power result exists for this experiment. The two cards were subsequently moved to Raider; the active owner's server launch has an unknown remote effect and the recovery hold remains.

Public candidate `380f8969c07514a83cb143f4ae53cada6b95fb13` is retained on isolated branch `fak/ticket-native-qwen35-device-bands-20261026`, not landed or published as a qualified runtime. [SOURCE-MANIFEST.json](SOURCE-MANIFEST.json) binds all 16 source files to committed Git bytes. Native `fak serve` integration is unchanged; the runnable candidate extends `cmd/pipelinegen`.

## Implemented

- Load only each stage's GGUF layer window, with matching memory census and explicit embedding/head ownership. Preserve packed Q2/Q6 storage instead of silently expanding everything.
- Keep KV and GDN state local to each stage; enforce absolute positions and legal hybrid cuts. Transfer raw activations between independent CUDA processes, avoiding tensor collectives.
- Check each card's own pool with headroom and check concurrent host demand separately. Aggregate 32 GB is not treated as one device allocation.
- Enable opt-in chunked prefill: one bounded handoff slot allows A's next chunk to dispatch while B handles the previous chunk. Device overlap and performance still require measurement.
- Bound startup, framing, failure teardown and child waits. Unknown cleanup returns an error; failed calls do not enqueue a graceful close behind unresolved work.

EP does not apply to this dense model. Merely building NCCL would not wire the missing tensor-parallel forward path; this candidate uses layer stages.

## Executed software evidence

[VERIFICATION.json](VERIFICATION.json) and [raw evidence](raw) retain the scope and counts:

- Two real CPU child processes load a nonzero eight-layer Qwen fixture as 4/4 stages and match monolithic prefill/decode logits, eight generated tokens and reset, including chunked dispatch. This is software evidence, not a two-GPU witness.
- Final CLI and race checks each passed 30 test events. A fresh normal CLI check also passed after the final path/shape cleanup.
- GGUF tests passed 917 events with one proven baseline failure excluded. Model remainder passed 10,373 events with two proven baseline failures excluded; compute's initial full package passed.
- All three excluded failure classes reproduced twice at the unchanged base, with byte-identical test sources. The initial model package also exhausted its 300-second deadline; its remainder passed in 301.204 seconds under a 600-second deadline. No MTP hang is established.

## Physical validation still required

The intended invocation, **not executed**, is a CUDA-built `pipelinegen` with `CUDA_VISIBLE_DEVICES=-1` on the coordinator, the two exact GPU UUIDs in `-devices`, the pinned GGUF in `-gguf`, `-backend cuda`, `-context 2048`, a legal `-cut`, and `-prefill-chunk 0` versus a bounded positive chunk. Token IDs, output cap, repetitions and profiling state must be pinned. The coordinator mask matters because the existing CUDA package initializes its backend before CLI parsing.

Use the exact pure-Q2 file SHA256 `a5929094ff5df70b5a16e91df35cce34ac192d2c29b0ffd8381f6b5356bb6e26`. After recovery, prove process/UUID placement, actual per-card fit, 27B correctness, selected kernels and end-to-end prompt/decode time; compare against a same-file llama.cpp layer split. Prior Anthony/Halo rows have different envelopes and cannot provide this result.

## Blockers and resume

The concurrency owner retains responsibility for the unknown Raider launch. Actual reconciliation and terminal cleanup/release are required before fresh grants; elapsed time is not a release. This task started no GPU workload and acquired no hardware grants.

Two local CUDA toolchain-preparation proposals were actually refused by Fak (`DEFAULT_DENY`, lease-admission, unknown write footprint). [WSL receipt](raw/admission-control-toolchain.json) and [native-check receipt](raw/admission-native-toolchain-check.json) both record `not_executed`. The native check also writes a receipt/cache/temp data, so a read-only hint would be inaccurate. No self-approval, renamed-tool proxy or denied-call replay occurred.

Resume through the recovery coordinator's reviewed closed CUDA executor when available: freeze this candidate source, compile `./cmd/pipelinegen` without a GPU smoke, then run the bounded physical canary and measurements under fresh leases. [STATE.json](STATE.json) records the handoff. Runtime publication remains held until qualification.
