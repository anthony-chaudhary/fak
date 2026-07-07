---
title: "Qwen3.6-27B GDN-hybrid Metal prefill — per-stage isolation split (diagnostic shipped; M3 Pro wall-clock ladder operator-gated) (2026-07-07)"
description: "Issue #2725: the claude-mac-fak flagship names a 10-15 min first-turn prefill on an M3 Pro (Qwen3.6-27B via Metal). This ships the prefill-only isolation split #67 pioneered for decode, refined to name the hybrid-specific suspects (GDN recurrence, full-attention, QK-norm) separately instead of one 'rest' bucket, plus env-gated stage skips (FAK_PREFILL_NO_GDN / FAK_PREFILL_NO_ATTN) to cross-check on-device serialization. Correctness is host-witnessed (logit cosine 0.999999 + greedy token-parity vs the CPU Q8 template). The before/after WALL-CLOCK ladder is deferred to the M3 Pro bench node — no prefill time was run, estimated, or fabricated on this (Windows, no Metal) host."
date: 2026-07-07
---

# Qwen3.6-27B GDN-hybrid Metal prefill — isolation split (issue #2725)

**Status:** host-tractable isolation-split diagnostic + correctness witness **SHIPPED** on `main`
(logit cosine `0.999999`, greedy token-parity, gated, default path untouched).
**Blocked:** the before/after wall-clock prefill ladder is deferred to the **M3 Pro Metal bench
node** (#67/#119/#1242). This host is Windows — no Metal runtime. **No prefill time was run,
estimated, or fabricated here.**

## What this issue is (and is not)

`claude-mac-fak` runs Qwen3.6-27B (GDN-hybrid) via Metal with a first local turn of **10–15 min
prefill on an M3 Pro**, single-stream — the fact #2691 fences as "no SOTA-speed framing." That
prefill gap has no owner: #977's Metal children (#71/#65/#934) target **decode** parity, not
prefill; #67 (closed) proved the isolation-split *method* closes this class of gap for decode
(0.56× → 0.988× of llama.cpp-Metal) but explicitly declined the hybrid recurrence/QK-norm path.

This is **prefill** time (compute-bound GEMM over the whole prompt), **not** decode tok/s. The
same isolation-split method #67 used for decode is applied here to the gap #977 does not name.

## The measured lever, and why the old profile could not find it

The hybrid Metal prefill runs the projection/MLP GEMMs on the GPU and keeps the recurrence,
attention, and norms on the CPU (`internal/model/metal_prefill_hybrid_core.go`,
`prefillQwen35HybridViaMM`). The pre-existing `FAK_QPROFILE` split was **2-way** — `gemm+roundtrip`
vs one `rest(recurrence/attn/norm)` bucket. That bucket lumps the three hybrid-specific suspects
#2725 asks about (the Gated-DeltaNet recurrence, the full-attention body, the QK-norm pass) into
one number, so it **cannot** answer "does the GDN recurrence or QK-norm force prefill onto an
unoptimized fallback." It could only say "not the GEMM" — it could not name which CPU stage is the
serial wall.

## What shipped (compute lane, host-witnessed)

1. **Per-stage isolation split.** `FAK_QPROFILE` now decomposes the CPU remainder into the named
   stages, each timed as its wall time minus the GEMM roundtrip that ran inside it:

   ```
   [metalprof-hybrid P=N] total=…  gemm+roundtrip=…  gdn-recurrence=…  full-attn=…  qk-norm=…  norm+act=… ms
   ```

   `gdn-recurrence` is the linear-attention (Gated-DeltaNet) layers' CPU cost; `full-attn` is the
   full-attention layers' CPU cost; `qk-norm` is the QK-norm pass surfaced as a sub-slice of
   `full-attn` (the #2725 suspect); `norm+act` is the RMSNorms + SwiGLU remainder. This is the
   diagnostic that names the serialized bottleneck rather than guessing — scope bullets 1 and 2.

2. **Env-gated stage skips** (mirroring #67's `FAK_DECODE_NO_ATTN` / `FAK_DECODE_MATMUL_ONLY`):
   `FAK_PREFILL_NO_GDN` drops the recurrence layers, `FAK_PREFILL_NO_ATTN` drops the full-attention
   layers; **both** set leaves only the GEMMs+norms (the matmul-only floor). On-device, an async GPU
   GEMM can overlap CPU work in a way a pure timer misattributes; skipping a stage and re-timing the
   wall reveals the true serialization the timer cannot. **Diagnostic only** — a skipped stage leaves
   its cache unfilled, so a skip run is for TIMING a bottleneck, never for a correct forward.

3. **Correctness pinned, host-independently.** `TestQwen35HybridViaMMMatchesCPUTemplate` asserts the
   whole prefill (logits + KV cache + linear-attn cache) matches the proven `prefillQwen35HybridQHidden`
   Q8 CPU template at **logit cosine ≥ 0.999999** (via `assertQuantLogitsClose`) **and greedy
   token-parity** (argmax). `TestQwen35HybridPrefillIsolationGatesZeroTheStage` witnesses that
   `FAK_PREFILL_NO_GDN` genuinely zeroes the `gdn-recurrence` stage and changes the forward (the gate
   is wired to the real body, not a no-op). Both run on any host — no Metal.

Gate: the split and skips are **off unless their env var is set**; the default forward is
byte-identical (the QK-norm pass is split from the bias pass but is numerically identical — every
token gets its bias before any QK-norm, the same order as the fused loop). `go test ./internal/model`
is green; `go build ./internal/model ./internal/compute` and `go vet ./internal/model` are clean.

## Before/after prefill-time ladder — the ladder shape, timings OPERATOR-GATED

Each rung names the diagnostic that would isolate it. The **wall-clock** cells are a measurement on
the M3 Pro Metal node and are **left blank here on purpose** (this host has no Metal). Do NOT fill
them from a decode number or an estimate.

| rung | diagnostic | what it isolates | P=256 | P=512 | P=1024 |
|---|---|---|---|---|---|
| 0 | baseline (`FAK_QPROFILE`) | current prefill wall, per-stage split | _M3 Pro_ | _M3 Pro_ | _M3 Pro_ |
| 1 | `FAK_PREFILL_NO_GDN` | wall with the GDN recurrence removed → its share | _M3 Pro_ | _M3 Pro_ | _M3 Pro_ |
| 2 | `FAK_PREFILL_NO_ATTN` | wall with full-attention removed → its share | _M3 Pro_ | _M3 Pro_ | _M3 Pro_ |
| 3 | both skips set | matmul-only floor (GEMMs + norms) | _M3 Pro_ | _M3 Pro_ | _M3 Pro_ |
| 4 | `qk-norm` sub-timer | QK-norm share within full-attn | _M3 Pro_ | _M3 Pro_ | _M3 Pro_ |

The **chunked-prefill sizing** suspect (scope bullet 3, `docs/serving/dual-track-serving-plan.md`
"[PARTIAL], ≤512 rectangular") is read off rung 0 at P=512 vs P=1024: if the ≤512 path and the
>512 path diverge in per-token wall beyond the O(P²) attention term, the claude-mac-fak turn is
falling onto the slower rectangular fallback above 512.

## The named blocker (do NOT fake this on a Metal-less host)

The done-condition's **wall-clock** before/after ladder is a measurement on an **M3 Pro** running a
clean GPU (stop the launchd `com.fak.qwen36-model` llama-server first, restore on EXIT — placement
law, #67's runbook). It cannot be witnessed on Windows. What IS witnessed here — the isolation-split
diagnostic itself, the env-gated skips, and the CPU-side correctness (cosine + token-parity) — is
the host-tractable half; the timing is the irreducibly Mac-gated residual.

## Reproduce (on `node-macos-a`, M3 Pro)

```bash
# clean GPU
launchctl bootout gui/$(id -u)/com.fak.qwen36-model   # restore after (EXIT trap)
G=~/.cache/fak-models/gguf/qwen3.6-27b-*-q8_0*.gguf
# rung 0 — per-stage split
FAK_QPROFILE=1 ./.bin/modelbench -gguf $G -prefill-sizes 256,512,1024 -lean
# rung 1/2/3 — isolation skips (TIMING ONLY; output is not correct under a skip)
FAK_QPROFILE=1 FAK_PREFILL_NO_GDN=1  ./.bin/modelbench -gguf $G -prefill-sizes 256,512,1024 -lean
FAK_QPROFILE=1 FAK_PREFILL_NO_ATTN=1 ./.bin/modelbench -gguf $G -prefill-sizes 256,512,1024 -lean
FAK_QPROFILE=1 FAK_PREFILL_NO_GDN=1 FAK_PREFILL_NO_ATTN=1 ./.bin/modelbench -gguf $G -prefill-sizes 256,512,1024 -lean
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fak.qwen36-model.plist
# correctness (any host, no Metal):
go test -run 'TestQwen35HybridViaMMMatchesCPUTemplate|TestQwen35HybridPrefillIsolationGatesZeroTheStage' ./internal/model
```

## Honesty fences

- **Prefill, not decode.** Every number this issue accepts is a prefill wall-clock; a decode tok/s
  does not answer it.
- **No fabricated timing.** The wall-clock ladder cells are blank pending the M3 Pro run. The shipped
  half is the diagnostic + the host-independent correctness witness, both green here.
- **Default path untouched.** The split, skips, and QK-norm sub-timer are env-gated / nil-guarded; a
  normal prefill is byte-identical and the correctness test proves it (cosine 0.999999).
- **Partial win is a valid outcome** (per the issue and #977's non-goals): this names the bottleneck
  measurement apparatus and pins correctness; it does not yet claim any backend is at prefill parity.
