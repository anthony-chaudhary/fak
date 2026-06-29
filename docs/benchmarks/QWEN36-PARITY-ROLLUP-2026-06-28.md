---
title: "Qwen3.6-27B vs llama.cpp parity — single-page rollup (2026-06-28)"
description: "One authoritative reconciliation of the scattered Qwen3.6-27B-vs-llama.cpp parity status: what is PROVEN (host-independent) vs what is `not yet` (Apple-Silicon / GPU / 27B-artifact gated), each gated witness with its exact one-command repro and the host capability it needs."
date: 2026-06-28
---

# Qwen3.6-27B vs llama.cpp — parity rollup (2026-06-28)

This page reconciles the scattered Qwen3.6-27B (`qwen35` arch, hybrid Gated-DeltaNet)
parity status into one document an auditor can read top-to-bottom. It does not introduce
a single new number: every figure traces to a file cited inline, and every gated row
names the host capability it needs and the exact one command that produces its witness.

**Provenance / honesty rule.** This rollup was assembled on a `windows/amd64`
orchestrator host with **no Apple Silicon, no NVIDIA GPU, and no 27B artifact**. Per
[`docs/proofs/00-METHOD.md`](../proofs/00-METHOD.md), every speed / GPU / 27B figure
below is a **recorded prior witness from a Mac (or AMD/Vulkan) node**, never re-measured
here. A SKIP is not a PASS; a gated item is never presented as run-here.

---

## 1 — The verdict, in one paragraph (two senses of "parity")

"Parity" means two different things and they have **different states**:

- **Correctness parity** (does fak's greedy decode reproduce llama.cpp's token stream?):
  **PROVEN at the architecture level** — the tiny `qwen3_5` HF fixture is bit-exact vs HF
  transformers (cosine 1.000000, max|Δ| ~4e-9, argmax parity at every position) — but
  **REFUTED at 27B scale**: on the real `Qwen3.6-27B.q4_k_m.gguf`, fak and llama.cpp match
  the first **two** tokens, then fak's argmax flips on a near-tie at **token 3**. This holds
  on *both* fak paths (GGUF→Q8 and resident-q4k), so it is diagnosed as a **kernel-numerics
  divergence at 27B scale on the hybrid GDN path** — not a Q8 round-trip artifact and not a
  bug in the reference path.
- **Speed parity** (tok/s vs the llama.cpp-Metal bar): **`not yet`**. fak's own M3 Pro engine
  runs the 27B end-to-end but its single-stream decode is **0.1 → 0.9 → 1.2 tok/s** along its
  three measured paths, still ~6× under the **7.29 tok/s** llama.cpp-Metal bar. The wall is
  diagnosed (per-call command-buffer launch overhead, not arithmetic) and the kernels are
  bit-correct; the closing levers are tracked but Apple-Silicon-gated.

The model **runs end-to-end in chat through fak's own in-kernel engine** on the M3 Pro
(no llama.cpp in the path) — that part is proven; what remains open is correctness *at
27B scale* and *speed*.

---

## 2 — Proven (host-independent — runs on a plain CPU box, no GPU / no 27B)

These are green on the orchestrator class of host (CPU-only, no weights). They are the
durable parity floor.

| What | Witness (test / artifact) | Result | Source |
|---|---|---|---|
| Architecture math bit-exact vs HF | `TestOptionalQwen35HybridOracleForwardMatchesHF` (tiny `qwen3_5` fixture, 3 GDN + 1 gated full-attn layer) | per-layer hidden-state cosine **1.000000**, max\|Δ\| **~4e-9**, **argmax parity** at every position | [`FAK-NATIVE-QWEN35-RESULTS.md`](FAK-NATIVE-QWEN35-RESULTS.md); [`QWEN36-PARITY-RESULTS.md`](QWEN36-PARITY-RESULTS.md) §"fak-native status" |
| Tokenizer byte-exact vs llama.cpp | `internal/tokenizer` oracle gate (#90) | byte-exact on the Qwen vocab + the 22-token ChatML smoke prompt | [`QWEN36-PARITY-RESULTS.md`](QWEN36-PARITY-RESULTS.md) |
| GGUF tensor mapping (tiny + real) | `TestQwen35GGUFConfigCanonicalizesHybridTensorsAndRunsForward`, `TestOptionalQwen35GGUFMapsEveryTensorName` | all 851 real-GGUF tensors map; hybrid knobs derived | [`FAK-NATIVE-QWEN35-RESULTS.md`](FAK-NATIVE-QWEN35-RESULTS.md) |
| Cached session == cacheless forward | `TestQwen35HybridSessionMatchesForwardAndPersistsState`, `TestQwen35HybridQuantTokenLoopPersistsState` | last-position logits match | [`FAK-NATIVE-QWEN35-RESULTS.md`](FAK-NATIVE-QWEN35-RESULTS.md) |
| #71 Metal-hybrid-prefill CPU orchestration | `TestQwen35HybridViaMMMatchesCPUTemplate` (drives `prefillQwen35HybridViaMM`) | logits + KV cache + linear-attn cache match the proven CPU template within ~1e-6 Q8 float-order drift; green on `windows/amd64`, `CGO_ENABLED=0` | [`experiments/qwen36/metal-hybrid-prefill-status-2026-06-28.md`](../../experiments/qwen36/metal-hybrid-prefill-status-2026-06-28.md) §2 |
| q4_k GEMM/GEMV dispatch bit-identical | `TestQ4KGemmMatchesMatRows`, `TestQ4KGemmInt8MatchesMatRowsInt8`, `TestQ4KMatRowsMatchesF32` | batched GEMM bit-identical to per-token decode GEMV (f32 + int8-SDOT) — the q4_k majority adds **zero** drift | [`experiments/qwen36/metal-q4k-device-gemm-status-2026-06-28.md`](../../experiments/qwen36/metal-q4k-device-gemm-status-2026-06-28.md) §3; [`…decode-gemv-status…`](../../experiments/qwen36/metal-q4k-decode-gemv-status-2026-06-28.md) §3 |
| #71 model-lane code LANDED on `main` | core `prefillQwen35HybridViaMM` (`c80d64fa`); Metal twin + gate + stub + `kv.go` dispatch (`5c065118`) | `dos commit-audit` `diff-witnessed` (`code_effect`) | [`…metal-hybrid-prefill-status…`](../../experiments/qwen36/metal-hybrid-prefill-status-2026-06-28.md) §1/§3 |

Build note for re-verifying any Go witness here: HEAD often doesn't build standalone (it
references uncommitted peer fields, e.g. an in-flight `forward.go`/`normWeights`
refactor); build the live tree with the peer pkg frozen at clean HEAD (`git archive HEAD`
into a scratch root). See `metal-hybrid-prefill-status-2026-06-28.md` §2.

---

## 3 — `not yet` (Apple-Silicon / GPU / 27B-artifact gated)

Each row needs a host this orchestrator does not have. The figures shown are **recorded
prior witnesses** from the named Mac/Vulkan node, never re-measured here; the repro is the
exact one command that re-confirms the gated half.

| # | Gated witness | Recorded prior result | Host needed | One-command repro |
|---|---|---|---|---|
| 1 | **27B greedy correctness parity** (close the token-3 drift to PROVEN) | REFUTED — fak `[248068,198,8160]` vs llama.cpp `[248068,198,90700]` (near-tie argmax flip at step 2) | M3 Pro + 27B GGUF + llama.cpp b9707 | `go run ./cmd/qwen35check -gguf ~/.cache/fak-models/gguf/Qwen3.6-27B.q4_k_m.gguf -n 3 -topk 5 -ids <22 ChatML ids> -expect 248068,198,90700` (exits ≠0 today) |
| 2 | **#71 Metal hybrid prefill — GPU f16 GEMM numerics** | code landed; GPU-numerics parity unverified | M3 Pro, `-tags fakmetal` (macOS Metal toolchain) | `go test ./internal/model -tags fakmetal -run Qwen35Hybrid -count=1` |
| 3 | **#63 on-device fak-Metal prefill tok/s** (post-#1085, clean) | warm prefill **2.6** @P=27 / **7.3** @P=940 (cold first 0.5); bar **51.55** @pp22 | M3 Pro, `-tags fakmetal`, `FAK_Q4K=1 FAK_METAL=1`, **no co-resident llama-server** (36 GiB swap rule) | `FAK_Q4K=1 FAK_METAL=1 FAK_QPROFILE=1 fakchat -gguf Qwen3.6-27B.q4_k_m.gguf -tok <dir> ...` → record `[metalprof-hybrid …]`; archived at `experiments/qwen36/metal-fak-q4k-post1085-m3pro-20260628.json` |
| 4 | **#68 decode GEMV on-device parity** | DONE in-thread (owner): `TestMetalQ4KGemvMatchesCPU` cosine **1.000000**; `TestMetalQ4KDecodeMatchesCPU` decode == CPU `[433 92 166 106]` | M3 Pro, `-tags fakmetal` | `go test ./internal/model -tags fakmetal -run 'MetalQ4K(Gemv\|Decode)' -count=1` |
| 5 | **#70 q4_k device GEMM matmul-only split** | code shipped; whole-path warm prefill 7.3 tok/s @P=940 (~7× under 51.55) | M3 Pro, `-tags fakmetal`, no co-resident llama-server | `go test ./internal/model -tags fakmetal -run MetalQ4K -count=1`; then `FAK_QPROFILE=1` pp22/long-prompt prefill |
| 6 | **#69 zero-copy residency + residency-win measure** | residency SHIPPED; `newBufferWithBytesNoCopy` upgrade + A/B win unmeasured | M3 Pro, `-tags fakmetal` (also needs a `(fak model)`-lane `FAK_METAL_REUPLOAD` baseline toggle first) | `go test ./internal/model -tags fakmetal -run MetalQ4K -count=1` after the toggle lands |
| 7 | **#67 end-to-end decode tok/s → 7.29 bar** | clean decode **1.2** tok/s (ratio 0.16×, perf-gate FAIL is the expected fail-closed state) | M3 Pro, `-tags fakmetal`, no co-resident llama-server | `python tools/qwen36_perf_gate.py --metal --min-ratio 0.5` (exit 1 = recorded gap) |
| 8 | **#65 GDN-recurrence on-device fraction** | DECIDED (CPU-hybrid, both phases — recurrence ≈0.5% of prefill); on-device capture pending | M3 Pro, `-tags fakmetal` | `FAK_QPROFILE=1` pp22 prefill → read `rest(recurrence/…)` vs `gemm+roundtrip` off `[metalprof-hybrid]` |

The single one-command Mac gate that drives rows 2–8 in sequence is being assembled at
`tools/qwen36_mac_parity_gate.sh` (sibling agent, this campaign).

---

## 4 — Decode-progression reconciliation (0.1 → 0.9 → 1.2 vs 7.29)

There is **one** "fak Qwen3.6-27B decode" number, not three rivals: it is a **measured
progression along three paths on one M3 Pro**, all single-stream / batch=1.
[`QWEN36-PARITY-RESULTS.md`](QWEN36-PARITY-RESULTS.md) is the **source of record** for the
full reconciliation table — do not duplicate it; this is the one-line summary:

| fak decode path | tok/s | what it measures |
|---|---:|---|
| GGUF→Q8 cached (CPU) | **0.1** | one cached decode token through the GGUF→Q8 round-trip |
| resident-q4k microbench (CPU) | **0.9** | raw q4_k blocks resident, scalar-f32 GEMV (~9× the Q8 path) |
| resident-Q4_K Metal (GPU) | **1.2** | int8-SDOT Metal decode GEMV; bit-correct (cosine 1.0) but launch-bound |

vs **llama.cpp-Metal 7.29 tok/s** (the bar). Why fak sits ~6× under it is **orchestration,
not arithmetic**: each decode token runs ~336 *separate* Metal command-buffer GEMVs, each
~360 µs launch/sync-bound on top of ~98 µs of bandwidth-limited work. The kernels are
correct (GEMV cosine 1.000000 vs CPU; greedy token-parity). The proof the lever works:
`BenchmarkMetalQ4KGemvBatch` runs 64 GEMVs in one command buffer at **5.2× faster/GEMV**
(89 GB/s, ~59% of device BW), projecting a one-command-buffer resident decode forward (#67)
to **~5.9 tok/s** (→ ~8 with a kernel pass) — right at the 7.29 bar. Full diagnosis:
[`docs/notes/MAC-QWEN36-27B-Q4K-METAL-PERF-DIAGNOSIS-2026-06-26.md`](../notes/MAC-QWEN36-27B-Q4K-METAL-PERF-DIAGNOSIS-2026-06-26.md).

---

## 5 — The correctness token-3 drift (summary + pointer)

On the exact 22-token ChatML smoke prompt, greedy / `temperature=0` / `top_k=1`:

| step | fak (in-kernel) | llama.cpp b9707 Metal | match |
|---|---|---|---|
| 0 | `248068` `<think>` | `248068` `<think>` | ✅ |
| 1 | `198` `\n` | `198` `\n` | ✅ |
| **2** | **`8160` `Here`** (logit 23.18; 2nd `90700` 21.43) | **`90700` `Thinking`** (logprob −0.547; 2nd `8160` −0.945) | ❌ |

Both engines surface the **same top-2 set `{8160, 90700}`** and rank it oppositely — a
**near-tie argmax flip** (~1.75 logits on fak's side, ~0.40 nats on llama.cpp's), the
signature of accumulated float drift reaching the decision boundary by token 3, not a gross
error. The drift survives the move from GGUF→Q8 to native resident-q4k weights on **both**
engines, which **disproves** the "Q8 round-trip quant artifact" hypothesis; combined with
the tiny-fixture HF bit-exactness (§2) and Qwen3.5-0.8B f32 semantic correctness through the
same arch path, it localizes to a **kernel-numerics divergence at 27B scale on the hybrid
GDN recurrence / mRoPE / partial-RoPE**. Recorded as **THEOREM 3 (REFUTED)** in the proof
ledger. Pinned artifacts:
`experiments/qwen36/native-gguf-q8-multitoken-parity-20260619.json` (fak) and
`experiments/qwen36/llamacpp-qwen36-multitoken-oracle-20260619.json` (oracle). Sources:
[`experiments/qwen36/QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md`](../../experiments/qwen36/QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md)
§1, [`QWEN36-PARITY-RESULTS.md`](QWEN36-PARITY-RESULTS.md) §"Token-3 drift RE-DIAGNOSED".

Deeper root-cause investigation (which GDN/RoPE op compounds the error):
`experiments/qwen36/token3-drift-investigation-2026-06-28.md` (sibling agent, this campaign).

---

## 6 — The one-command Mac gate

`tools/qwen36_mac_parity_gate.sh` (sibling agent, this campaign) is the single entry point a
Mac verify node runs to re-confirm the gated half of §3 in one go — the `-tags fakmetal`
build + GPU numerics parity + the clean (no co-resident llama-server) `FAK_QPROFILE` tok/s
captures, against the recorded 51.55 / 7.29 bars. Until it is green on a witnessed commit,
rows 1–8 of §3 stay `not yet`.

---

## 7 — Adjacent axes — explicitly NOT the single-stream kernel numbers above

Two other Qwen3.6-27B figure families circulate; they are a **different denominator** and
must not be quoted on the same line as the M3 Pro single-stream kernel rows:

- **8-GPU served throughput** (single-stream ≈59–93 tok/s, batched peak ≈820–1085
  completion tok/s) comes from **SGLang-serves + fak-adjudicates** on a datacenter GPU host,
  not fak's own M3 Pro engine — see [`QWEN36-27B-GPU-SERVER-RESULTS.md`](QWEN36-27B-GPU-SERVER-RESULTS.md).
- **AMD/Vulkan desktop** ([`QWEN36-AMD-VULKAN-RESULTS.md`](QWEN36-AMD-VULKAN-RESULTS.md))
  proves the model *loads and serves* on an RX 7600, but llama.cpp logs `fused Gated Delta
  Net (chunked) not supported, set to disabled` — so its absolute throughput is **not** an
  apples-to-apples GDN bar. Likewise **MLX is an invalid bar** for this arch (its hybrid
  cache silently recomputes the full context every turn) — never used here; see
  [`…PARITY-AND-MEASUREMENT-STATUS…`](../../experiments/qwen36/QWEN36-PARITY-AND-MEASUREMENT-STATUS-2026-06-20.md) §2.

The single measured **bar** for the M3 Pro lane is **llama.cpp b9707 Metal (`-ngl 99`):
prefill 51.55 tok/s, decode 7.29 tok/s, peak RSS ~24.5 GB**; CPU-only (`-ngl 0 -t 6`):
20.12 / 6.48. (Source: [`QWEN36-PARITY-RESULTS.md`](QWEN36-PARITY-RESULTS.md).)

---

_Rollup assembled 2026-06-28 on a win32 orchestrator (no Apple Silicon / no NVIDIA GPU / no
27B artifact). All speed/GPU/27B figures are recorded prior Mac/Vulkan-node witnesses, cited
inline; none re-measured here._
