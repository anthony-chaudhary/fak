# BENCHMARK AUTHORITY — Single Source of Truth

> **Why this exists.** This repo contains many benchmark results across different axes (raw throughput, reuse efficiency, session value-add, etc.). This document is the **authoritative index** of all committed benchmark claims, with traceability to source commits and artifact files. **Any number claimed elsewhere must trace back to an entry here.**

> **📋 Process:** See **[BENCHMARK-GOVERNANCE.md](BENCHMARK-GOVERNANCE.md)** for the DOS-centric process that creates, verifies, and publishes these claims. This file is the *what* (the numbers); Governance is the *how* (the discipline).

> **🏆 Presentation layer:** **[HERO-BENCHMARK-2026-06-21.md](HERO-BENCHMARK-2026-06-21.md)** is the frontier-lab-style *hero comparison* (v1) built **from** this authority — headline number, top-3 SOTA chart, top-10 leaderboard with fak bolded where it wins (and the two single-stream losses shown plainly). It claims no new numbers; every figure traces to a row below.

> **🧠 The *why*:** `WHY-REUSE-WINS-2026-06-21.md` (private companion — not published) (v1) argues — and stress-tests — *why* these reuse numbers matter more than the headline alone: reuse is a **different class** of optimization (work-elimination on the `N` axis, not work-acceleration on the `κ` axis), so it's **exact, training-free, and composes multiplicatively** on top of every per-token trick. Follows the v2 SOTA-only framing — leads with the absolute competitive number (**19.0 min vs 78 min = 4.1× less work**, conservative marginal 2.4–2.7×), shows **no naive-loop numbers**, and centers cross-agent reuse as the layer that is `fak`'s. No new numbers; fences where the "works across everything / no fine-tuning" framing is overstated (and where addressable reuse, [#228](https://github.com/anthony-chaudhary/fak/issues/228), widens it).

**Last updated:** 2026-06-21
**Status:** Living document — update when new model results ship

---

## Quick Reference: Primary Numbers

| Claim | Number | Model | Baseline | Commit | Artifact |
|---|---|---|---|---|---|
| **fak CPU Q8 single-stream vs llama.cpp CPU (M3 Pro)** — CANONICAL | **decode 0.58× (41.9 vs 71.9) · prefill@256 0.45× (247.2 vs 547)** | Qwen2.5-1.5B Q8, M3 Pro | llama.cpp CPU −ngl 0 (71.9 / 547) | `3448d7b` | `model-ladder/qwen25-1.5b-q8-cpu-parity-m3pro.json` ← **single source; read, don't hardcode** |
| **RadixAttention live speedup (model ladder)** | **4.58× → 6.95×** | SmolLM2-135M → Qwen2.5-1.5B Q8 | Full re-prefill | `92896a4` | `radixbench-*-agents-fresh-20260619.json` |
| RadixAttention token speedup | 7.50× | all four models (Q8) | Token count | `92896a4` | Same (`prefill_token_speedup`) |
| RadixAttention hit rate | 86.7% (FCFS 62.1% → cache-aware) | all four models (Q8) | Cache hits | `92896a4` | Same (100% of optimal) |
| **README headline: 50-turn × 5-agent reuse win** | **60.3× vs naive · 4.1× vs tuned** | Qwen2.5-1.5B Q8, T=50 A=5 P=2048 | Naive stateless / tuned per-agent KV | `2bbda6f` | `headline-qwen-50x5.json` |
| **Fleet 5-agent × 200-turn 7B in <10 min (MEASURED, M3 Pro)** | **8.2 min · 2.5× vs single-stream · ≥30× vs naive** | Qwen2.5-7B Q8, T=200 A=5 P=2048 D=20 R=12, M3 Pro llama.cpp Metal | 5 single-stream sessions / naive re-prefill | _this commit_ | `session/macbook-m3pro-7b-batched-{bench,ctx}.log` (measured 17.41/44/392 t/s) + `fleet-5x200-7b-projection-20260622.json` + `FLEET-5X200-7B-10MIN-RESULTS.md`. Batched ≈ a tuned `llama-server --parallel`; fak's add is per-agent KV ownership + safety floor, not raw t/s |
| **Session value-add (high-T ladder)** | **24.9× → 139.3×** | SmolLM2-135M Q8, T=64 → T=512 | Naive stateless | `92896a4` | `highT-smollm2-135m-*-fresh-20260619.json` |
| Session value-add (1.5B "realistic model") | 7.2× → 10.0× | Qwen2.5-1.5B Q8, T=8 → T=16 | Naive stateless | `92896a4` | `smoke-qwen2.5-1.5b-T8-16-fresh-20260619.json` |
| ~~Session value-add 11.2–14.5× (SmolLM2, P=512)~~ | ❌ STALE | SmolLM2-135M Q8 | Naive stateless | `5b0f40d` | superseded — see F1 below |
| Session value-add (SmolLM2 P=512, re-measured) | **5.3–7.4×** | SmolLM2-135M Q8 | Naive stateless | `885ae8a` | `benchmark-run-opencode-20260619/sessionbench-smollm2-135m-q8-authority.json` |
| Qwen2.5-7B fak decode | 8.7 tok/s | Qwen2.5-7B Q8 | llama.cpp Metal 17.27 tok/s | `34c74f4` | `model-ladder/modelbench-qwen25-7b-q8.json` |
| Qwen2.5-7B fak/llama.cpp ratio | 0.50× decode / 0.083× prefill | Qwen2.5-7B Q8 | llama.cpp Metal | `34c74f4` | `QWEN25-7B-RESULTS.md` |
| Qwen2.5-7B greedy parity | ✅ full 7-token match | Qwen2.5-7B Q8 | llama.cpp ("2+2 is 4.") | `34c74f4` | `QWEN25-7B-RESULTS.md` |
| Qwen3.5-0.8B hybrid-GDN runs in fak | ✅ coherent ("pong") | Qwen3.5-0.8B f32 | instruction-following | `6a376b8` | `QWEN35-0.8B-RESULTS.md` |
| Qwen3.6-27B resident-q4k decode | 0.9 tok/s | Qwen3.6-27B q4_k_m | llama.cpp Metal 7.29 tok/s | `1698eff` | `model-ladder/qwen36-perf-gate-m3-20260619.json` |
| Qwen3.6-27B fak/llama.cpp ratio | 0.12× decode / 0.01× prefill | Qwen3.6-27B q4_k_m | llama.cpp Metal | `1698eff` | `model-ladder/qwen36-perf-gate-m3-20260619.md` |
| Qwen3.6-27B token parity | 2-token match (drift @3) | Qwen3.6-27B q4_k_m | llama.cpp oracle | `d03be46` | `model-ladder/qwen36-resident-q4k-parity-20260619.json` |
| Qwen3.6-27B surface smoke | 4/4 surfaces PASS | Qwen3.6-27B (served) | agent/gateway/mcp/dogfood | `8a0f5bc` | `model-ladder/qwen36-surfaces-dogfood-opencode-20260619.json` |
| Synthetic model live ratio | 1.64× | 64h/4L wiring | Full re-prefill | `a200c3d` | `radixbench-synthetic.json` |
| **GPU Q8 decode (Vulkan, RX 7600)** | **24.6 tok/s · 1.49× vs GPU f32** | SmolLM2-135M Q8 | Same forward, f32 weights on GPU | `60db592` | `q8gpu-smollm2-135m-{gpu-q8,gpu-f32}-20260619.json` |
| **GPU/CPU Q8 decode crossover** | **CPU lead 7.2× (135M) → 1.16× (1.5B)** | SmolLM2-135M → Qwen2.5-1.5B Q8 | CPU Q8 (legacy) | `7bf666b` | `crossover-qwen2.5-1.5b-{gpu,cpu}-q8-20260619.json` |
| Synthetic model live ratio | 1.64× | 64h/4L wiring | Full re-prefill | `a200c3d` | `radixbench-synthetic.json` |
| **GPU Q8 decode (Vulkan, RX 7600)** | **24.6 tok/s · 1.49× vs GPU f32** | SmolLM2-135M Q8 | Same forward, f32 weights on GPU | `60db592` | `q8gpu-smollm2-135m-{gpu-q8,gpu-f32}-20260619.json` |
| **GPU/CPU Q8 decode crossover** | **CPU lead 7.2× (135M) → 1.16× (1.5B)** | SmolLM2-135M → Qwen2.5-1.5B Q8 | CPU Q8 (legacy) | `7bf666b` | `crossover-qwen2.5-1.5b-{gpu,cpu}-q8-20260619.json` |
| **GPU decode parity (reusable CUDA graph, RTX 4070)** — README headline | **~120 tok/s (119–120, f32) · parity with llama.cpp Q8_0** | SmolLM2-135M, RTX 4070 Laptop sm_89 / WSL2 (gated `FAK_CUDA_GRAPH=1`) | llama.cpp Q8_0 (120 ± 15, `-ngl 99`) | `1029e37` | `GPU.md` §3b + `LLAMACPP-HEADTOHEAD-RESULTS.md` (on-box bench/test witness; reproduce: `FAK_CUDA_GRAPH=1 go run -tags cuda ./cmd/modelbench -dir internal/model/.cache/smollm2-135m -backend cuda`) |
| **Pure-kernel decide latency (M3 Pro)** | **362 ns** allow · 560–605 ns w/ ArgPredicates | syscall/adjudicator Decide | per-call decision | `bcad56e` | `experiments/mac-m3pro-kernel-20260620/kernel-latency-mac-m3pro-20260620.json` |
| **Pure-kernel admission latency (M3 Pro)** | **1.8–14 µs** scan · 3.3–15.8 µs Admit · 29–87 µs chain | ctxmmu / normgate+ctxmmu | per-result admission (cited "~1,300 ns" = cheapest scan layer only) | `bcad56e` | same (`MAC-M3PRO-KERNEL-BENCH-2026-06-20.md`) |
| **Syscall boundary tax (M3 Pro, refreshed)** | **~2,849×** in-process vs spawned `fak hook` | in-process adjudication | process-per-decide baseline (n=100) | `bcad56e` | `report.json` + `experiments/mac-m3pro-kernel-20260620/report.json` |
| **Causal invalidation-on-external-write** | **PASS · max\|Δ\|=0** (1 evicted, sibling warm, re-admit refused) | vDSO `Revoke` + cachemeta external-invalidation | blunt world-flush / stale serve | `0fc39aa` | `experiments/causal-invalidation-20260620/causalbench-witness-20260620.json` |

> **The model-ladder thesis.** Live wall-clock ratio climbs toward the deterministic
> 7.50× token-speedup ceiling as per-token compute grows (135M 4.58× → 360M 5.40× →
> 0.5B 6.20× → 1.5B 6.95×). This confirms that the residual gap below 7.50× is
> clone/memcpy overhead that becomes negligible on larger models — not an
> architectural limit. The deterministic metrics (token speedup, hit rate) are
> hardware-independent and reproduce the committed JSON exactly; only the live
> wall-clocks are single-box (within-run ratios authoritative per
> [BENCHMARK-GOVERNANCE.md](BENCHMARK-GOVERNANCE.md) regime rules).

---

## Pure-kernel latency — Apple M3 Pro (2026-06-20)

**Date:** 2026-06-20
**Commit:** `bcad56e`
**Files:** `fak/experiments/mac-m3pro-kernel-20260620/kernel-latency-mac-m3pro-20260620.json`, `fak/MAC-M3PRO-KERNEL-BENCH-2026-06-20.md`
**Machine:** Mac15,7 — Apple M3 Pro, 12 core, arm64, darwin, go1.26.0. Medians of count=8 trials on an idle box.

### What this adds (and why)

The Authority's model-bench rows left the **pure-kernel latency stack** uncommitted: the
syscall bench (`report.json`) was the one pure-kernel artifact and was explicitly "narrow",
and a "~1,300 ns" admission figure cited in `DISAGGREGATED-AGENT-MEMORY.md` and
`MEMORY-LAYERS-EXPLAINER.md` had no committed artifact. This pass witnesses the stack via
`go test -bench` (the most reproducible form) so every cited number traces to a committed
artifact + reproduction command. Full decomposition and honest fences in the results doc.

### Results

| Layer | p50 ns/op | B/op | allocs/op | verdict |
|---|---:|---:|---:|---|
| **Decide** (canonical allow) | **362** | 256 | 5 | ALLOW |
| Decide w/ ArgPredicates (0→2000 unrelated) | 560 → 605 | 600 | 14 | — |
| **ScreenBytes** scan — secret (regex) | **1,812** | 0 | 0 | caught |
| ScreenBytes scan — benign (full) | 4,482 | 128 | 2 | allow |
| ScreenBytes scan — injection (nested) | 14,062 | 417 | 2 | caught |
| **Admit** (full gate, +page-out) — secret | 3,337 | 2,022 | 26 | QUARANTINE |
| Admit — injection | 15,799 | 2,662 | 28 | QUARANTINE |
| **AdmitChain** (normgate+ctxmmu) — benign | 29,171 | 1,662 | 25 | ALLOW |
| AdmitChain — injection | 87,056 | 4,854 | 38 | QUARANTINE |

Plus the refreshed syscall A/B: in-process p50 **2,427 ns** vs spawned `fak hook` p50
**6.913 ms** (n=100) → **~2,849×** boundary tax, `gate_primary=pass`.

### The honest finding on the cited figure

The "~1,300 ns" is the **narrow reading** — the cheapest `ScreenBytes` path (secret regex)
measures ~1.8 µs here, same order, **not fabricated**. But it names only one layer: the
general scan is 4.5–14 µs, the full `Admit` (with the page-out side-effect) is 3.3–15.8 µs,
and the full normgate+ctxmmu chain the kernel `Reap` runs is 29–87 µs. The single cited
number undersells the composed path by ~an order of magnitude on the worst payload; the
decomposition above replaces it. (Governance rule #4: the old figure is marked, not
silently removed.)

### Verification

- New `internal/ctxmmu/bench_test.go` compiles + vets clean; `go test ./internal/ctxmmu
  ./internal/adjudicator` → PASS (existing ctxmmu tests unaffected by the normgate
  registration the chain bench enables).
- `dos_commit_audit bcad56e` → **ABSTAIN** (the subject is a witness/documentation claim, not
  a falsifiable code claim; the diff is nonetheless real — it lists `bench_test.go` + the
  committed JSON artifacts). The load-bearing witness is `dos verify fak benchmark` below.
- `dos verify fak benchmark` → **SHIPPED** (`bcad56e`, rung `trailer` — the `(fak benchmark)`
  stamp binds the commit as a unit of benchmark work, confirmed by evidence, not self-report).

---

## Causal invalidation-on-external-write (2026-06-20) — the CPU-only strategic witness

**Date:** 2026-06-20
**Commit:** `0fc39aa`
**File:** `fak/experiments/causal-invalidation-20260620/causalbench-witness-20260620.json`
**Reproduce:** `go run ./cmd/causalbench -selfcheck` (zero files, exits non-zero on any violation)

### What this witnesses (and why it is the cheapest strategic proof)

This is matrix row 6 of `PLAN-cloud-neocloud-rightsizing-2026-06-20.md` — the one
genuinely **net-new** strategic concept with **no hardware dependency** ($0, CPU-only,
unblocked). It proves the property `STRATEGIC-TIMING-2026-06-20.md` ranks #3: an external
write makes a cached read stale, and the system **itself** discovers *which* cached reads
depended on the now-stale world-state and evicts exactly those, byte-exact, refusing
re-admission. It is the causal sibling of the provable-deletion witness (`cmd/deletioncert`,
row 5): deletion evicts a span an operator *chose*; this evicts the reads an external write
*caused* to go stale — the MESI-invalidate analogue in the integrity direction.

The kernel mechanism was already shipped (`cachemeta.PlanExternalInvalidations`,
`vdso.Revoke`, `internal/gateway/coherence.go` wiring it onto live `fak serve` traffic).
What was missing — and what this adds — is a single self-checking end-to-end witness that
binds the whole chain, the artifact this row anchors. It runs on the **real process-global
`vdso.Default`** (the same `Lookup`/`Emit`/`Revoke` path live traffic uses), no model and
no weights, because the property is structural over cache identity + the witness ledger,
not numeric.

### The chain it proves (every row an asserted invariant; the demo exits non-zero otherwise)

| Invariant | Artifact field | Value |
|---|---|---|
| Two reads admitted under two external witnesses serve byte-exact from cache | `w1_hit_before_write` / `w2_hit_before_write` | true / true |
| Cached bytes equal a fresh engine call (a hit *is* a fresh call) | `w1_served_byte_exact` | true |
| External write refutes one witness → **exactly** the dependent read evicted | `w1_evicted_by_write` | **1** (targeted, not a flush) |
| The sibling under the unrefuted witness stays byte-identical across the write | `w2_byte_identical_across` | true (**max\|Δ\|=0**) |
| The refuted read now misses → goes to the engine, fresh (no stale serve) | `w1_miss_after_write` | true |
| A re-fill under the refuted witness does **not** repopulate (CAS can't resurrect it) | `w1_readmission_refused` | true |
| Refuting an unrelated witness evicts **0** local entries | `unrelated_witness_evicts` | 0 |
| The refutation is broadcast on the coherence bus (cross-agent propagation) | `coherence_broadcast_fired` | true |
| The integrity clock advances on refutation | `trust_epoch_advanced` | true |

### Honesty fences

- **This is a containment/coherence witness, not a throughput number.** It proves the
  causal-eviction *property* holds byte-exact on the real kernel path; it says nothing
  about tok/s or scale. Pool-scale behaviour under many concurrent agents is row 15 of the
  right-sizing plan and remains `[DEFERRED]` / projected.
- **Structural, not numeric.** Like `cmd/deletioncert`, it uses inline payloads and the
  witness ledger, so `max|Δ|=0` is an exact byte comparison of served payloads, not an
  approximate tolerance. No model is loaded; the claim is about cache identity, which is
  hardware-independent and reproduces the committed JSON exactly.
- **The witness is not keyed into the tier-2 key yet** (per `revoke.go`'s own honesty
  note): two agents reading under *different* witnesses still share by `(tool,args,worldVer)`.
  This witnesses the revocation axis (C4 causal-consumer eviction), which is the
  load-bearing half; witness-keying is the natural follow-on.

### Verification

- `go run ./cmd/causalbench -selfcheck` → exit 0 (all 12 guarded invariants hold — the
  9-row table above is the headline subset); `main_test.go` guards the same chain via the
  portable `go test ./cmd/causalbench/` → `ok cmd/causalbench` (on Windows: `.\fak\test.ps1`).
- `dos_commit_audit 0fc39aa` binds the result commit (diff-witnessed: the demo, its test,
  and the committed JSON artifact).
- The number is a correctness verdict (PASS / `max|Δ|=0`), not a wall-clock — hardware-
  independent and re-derivable from the artifact alone.

---

## README Headline — 50-turn × 5-agent Qwen2.5-1.5B (the number on the front page)

**Date:** 2026-06-19
**Commit:** `2bbda6f`
**File:** `fak/experiments/session/headline-qwen-50x5.json`
**Chart:** `fak/experiments/session/chart1-headline-walltime.svg`

This is the number a first-time visitor sees in README §1: *"On a realistic 50-turn ×
5-agent run (Apple M3 Pro, Qwen2.5-1.5B), fak did in ~19 minutes what the naive loop
needs an estimated ~19 hours."* Every figure in that sentence traces here.

### Shape & arms

`T=50 agents=5 prefix=2048 decode=32 result=64`, Qwen2.5-1.5B-Instruct Q8 (lean
quantize-at-load). Three arms over the **same Q8 forward pass**: A naive-stateless, B
per-agent-KV tuned, C fak fused (prefix prefilled once, cloned into the agents, batched
decode).

### Results (from the artifact)

| Metric | Value | Artifact field |
|---|---|---|
| **Reuse win vs naive** | **60.3×** | `net_value_add_vs_naive=60.346` |
| **Reuse win vs tuned warm-cache** | **4.1×** | `net_value_add_vs_tuned=4.125` |
| Arm A naive total | ~19.1 h | `arm_A_naive_stateless.total_ms=68,726,015` |
| Arm C fak total | ~19.0 min | `arm_C_fak_fused.total_ms=1,138,871` |
| Exact prefill-token ratio A/C | 62.0× | `prefill_tokens.a_over_c=62.05` |
| Turn-tax A/B | 14.6× | `turn_tax_A_over_B=14.63` |

### Honesty fences (matching the README's own)

- The **60.3×** is **vs the naive loop** (re-send the whole growing context every turn).
  Vs a *tuned* warm-cache stack the honest gain is **4.1×** — a few-fold, as stated.
- Arm A's ~19h is **modeled** from `prefillCost(L)` sampled at six lengths
  (`prefill_model.lens/ms` in the artifact), not run live (it would take ~19h).
  The model is **validated within ~0.4%**: `live_validate.anchored_computed_over_live
  = 1.0039` (a reduced live shape confirms the projection). The README's "within ~1%"
  is conservative against this.
- Arms B and C run **fully live** (attention growth captured); arm A's decode is set
  byte-identical to arm B's live decode. Disclosed in the artifact `methodology` field.

### Verification

- `dos_commit_audit 2bbda6f` binds the result commit.
- Bit-identity gates green (`TestBatchedDecodeMatchesSerial`,
  `TestBatchFromPrefixMatchesIndependentPrefill`): the three arms emit identical tokens,
  so the win is reuse, not a numerics shortcut.

### F1 — tombstone note (2026-06-19, Governance rule #4)

The old SmolLM2 session cells **11.2×/14.5×** (P=512, T=8/16, A=4) do **not reproduce** on the
current kernel: re-measured as **5.3×/7.4×**. Root cause: commits `70a2cab` (Q8 prefill softcap),
`6e5fda3` (SEAM-0 decode fold), `eb9a2e5` (q8 scratch reuse) between the `5b0f40d` measurement and
HEAD made the Q8 prefill ~2× faster, so computed arm A (naive re-prefill) got cheaper and A/C
shrank. The fak arm C still matches (12.0s/28.2s old vs 11.2s/24.4s re-measured). The old number
shrank **because fak got faster at the prefill arm A depends on**, not from any regression. Full
diagnosis: `benchmark-run-opencode-20260619/BENCHMARK-RUN-OPENCODE-20260619.md` finding F1.

---

## RadixAttention Results (SmolLM2-135M Q8)

**Date:** 2026-06-18
**Commit:** `a200c3d`
**File:** `fak/experiments/radixattention/radixbench-smollm2-135m-q8.json`

### What This Measures

Compares **baseline** (full re-prefill per request) vs **radix** (automatic prefix-cached KV reuse using the same algorithm as SGLang's RadixAttention paper).

### Workload: Agents

- **Shape:** 5 agents × 6 turns = 30 requests
- **System prefix:** 128 tokens (shared across all agents)
- **Per-turn step:** 24 tokens
- **Model:** SmolLM2-135M Q8_0 (30 layers, real checkpoint)

### Results

| Metric | Baseline | Radix | Speedup/Improvement |
|---|---|---|---|---|
| **Wall-clock** | 240,994 ms (~241s) | 49,452 ms (~49s) | **4.87×** |
| **Tokens computed** | 6,360 | 848 | **7.50×** fewer |
| **Cache hit rate** | 0% | 86.7% | Matches SGLang band (50-99%) |

### Verification

- `dos_commit_audit a200c3d` → **OK** (diff-witnessed)
- Committed JSON artifact exists and is readable
- Token counts are exact integers (hardware-independent)

### Why Wall-Clock (4.87×) < Token Speedup (7.50×)

On the synthetic 64-hidden/4-layer wiring model, the memcpy cost of cloning cached KV masks the compute savings (1.64× live ratio). On SmolLM2-135M, per-token compute dominates memcpy, so live speedup approaches the theoretical token figure (4.87×).

**Both results are committed and real** — they document different regimes.

> **Superseded as the headline by the 2026-06-19 model ladder below.** The single
> 135M point (`a200c3d`, contended 4.87×) remains a real committed measurement; the
> fresh ladder re-runs it at 4.58× (reps3, lightly contended) and extends it across
> three more models. Cite the ladder for the release; this row stays as provenance.

---

## RadixAttention Model Ladder (2026-06-19) — climbs to the token-speedup ceiling

**Date:** 2026-06-19
**Commit:** `92896a4`
**Files:** `fak/experiments/radixattention/radixbench-{smollm2-135m,smollm2-360m,qwen2.5-0.5b,qwen2.5-1.5b}-q8-agents-fresh-20260619.json`

### What This Adds

The same RadixAttention `agents` workload (5 agents × 6 turns, 128-token shared
system prefix, 24-token per-turn step) run across four real Q8 checkpoints. The
**deterministic** metrics (token speedup, hit rate, FCFS→cache-aware recovery) are
byte-identical across all four (model-independent); the **live wall-clock** ratio is
the one that moves, and it climbs monotonically toward the 7.50× token ceiling as the
model grows.

### Results — `agents` workload

| Model | Live wall-clock | Token speedup | Hit rate (FCFS → cache-aware) | Artifact (`live_prefill_speedup`) |
|---|---|---|---|---|
| SmolLM2-135M (30L) | **4.58×** | 7.50× | 62.1% → 86.7% (100% of optimal) | `radixbench-smollm2-135m-q8-agents-fresh-20260619.json` |
| SmolLM2-360M (32L) | **5.40×** | 7.50× | 62.1% → 86.7% | `radixbench-smollm2-360m-q8-agents-fresh-20260619.json` |
| Qwen2.5-0.5B (24L) | **6.20×** | 7.50× | 62.1% → 86.7% | `radixbench-qwen2.5-0.5b-q8-agents-fresh-20260619.json` |
| Qwen2.5-1.5B (28L) | **6.95×** | 7.50× | 62.1% → 86.7% | `radixbench-qwen2.5-1.5b-q8-agents-fresh-20260619.json` |

Deterministic hit rates reproduce committed `a200c3d` exactly: few-shot 88.2%,
multi-turn-chat 79.5%, tree-of-thought 77.2%, agents 86.7%. Policy-eviction witness
green on every run.

### Verification

- Each row's `live_prefill_speedup` read directly from its committed JSON (verified
  2026-06-19: 4.581 / 5.40 / 6.20 / 6.951 → rounded above).
- `internal/radixkv` split-reuse == recompute (max|Δ|=0) → **PASS** (numerics are
  reuse, not a shortcut).
- Token counts (`prefill_token_speedup=7.5`, `radix_computed_tokens=848`) are exact
  integers, hardware-independent.
- **Cross-platform reproduction (2026-06-19):** the 135M `agents` deterministic fields
  reproduce **bit-for-bit on Windows x86_64** (hit 86.7%, token 7.50×, reused 5512,
  computed 848) vs the Mac M3 arm64 committed artifact; the live ratio moves (2.60× on
  x86 vs 4.58× on Mac) exactly as the small-model clone-overhead thesis predicts. See
  [`experiments/radixattention/CROSS-PLATFORM-REPRO-20260619.md`](experiments/radixattention/CROSS-PLATFORM-REPRO-20260619.md).

---

## Session Value-Stack High-T Ladder (2026-06-19) — the O(T²)→O(T) contrast

**Date:** 2026-06-19
**Commit:** `92896a4`
**Files:** `fak/experiments/session/highT-smollm2-135m-{64-128-256,512}-fresh-20260619.json`

### What This Adds

The session value-stack (A=naive-stateless, B=per-agent-KV tuned, C=fak fused) pushed
to high turn counts on SmolLM2-135M (P=512, D=4, R=8, C=2) to expose the naive arm's
O(T²) re-prefill signature against fak's near-linear curve.

### Results

| T | A naive | B tuned | C fak | **A/C vs naive** | turn-tax A/B | exact prefill-tok A/C |
|---|---|---|---|---|---|---|
| 64  | 268.1s | 14.3s | 10.8s | **24.9×** | 18.7× | 74.9× |
| 128 | 908.8s | 30.7s | 23.0s | **39.5×** | 29.6× | 128.2× |
| 256 | 3982.1s | 74.8s | 54.4s | **73.2×** | 53.2× | 227.7× |
| 512 | 20424.4s (~5.7h) | 211.5s | 146.6s | **139.3×** | 96.6× | 421.7× |

The naive arm A explodes ~4× per T-doubling (268→909→3982s) — the O(T²) re-prefill
signature — while B and C stay near-linear.

### Honest methodology (carried from the artifact)

Arms **B and C run end-to-end LIVE** (attention growth captured). Arm **A's prefill
is modeled** from `prefillCost(L)` measured at sampled lengths (the O(L²)
prefill-attention captured, summed over the exact per-turn contexts), because running
A fully live at T=512 would take ~5.7h per cell; arm A's decode is set byte-identical
to arm B's live decode. The `validate` shape runs arm A fully live to confirm the
model. This is disclosed in each JSON's `methodology` field.

### Verification

- T=512 cell read from artifact: `net_value_add_vs_naive=139.278`,
  `turn_tax_A_over_B=96.564`, `prefill_tokens.a_over_c=421.716`.
- Bit-identity gates green: `TestBatchedDecodeMatchesSerial`,
  `TestBatchFromPrefixMatchesIndependentPrefill` (arms produce identical tokens).

---

## GPU Q8 Throughput — Vulkan on the AMD RX 7600 (2026-06-19)

**Date:** 2026-06-19
**Commit:** `60db592` (path unblocked by `84c2e6c`)
**Files:** `fak/experiments/gpu/q8gpu-smollm2-135m-{gpu-q8,gpu-f32,cpu-q8}-20260619.json`
**Doc:** `fak/experiments/gpu/VULKAN-Q8-RX7600-20260619.md`

### What This Adds

The first committed Q8-on-GPU throughput numbers from the `modelbench` harness. Until
`84c2e6c`, `modelbench -backend vulkan -quant` hard-refused ("compute HAL sessions are
f32-only today") even though the Q8 weight-upload + device-GEMM path was fully wired in
`internal/model/hal.go`. Three arms over the **same SmolLM2-135M Q8 forward pass** on the
real RX 7600 (Vulkan 1.4.349, native Windows), 64 decode steps / 3 reps.

### Results

| arm | decode tok/s | prefill P=16 → 512 | artifact |
|---|---:|---:|---|
| **gpu-q8** | **24.6** | 15.6 → 24.8 | `q8gpu-smollm2-135m-gpu-q8-20260619.json` |
| gpu-f32 | 16.5 | 12.6 → 18.7 | `q8gpu-smollm2-135m-gpu-f32-20260619.json` |
| cpu-q8 | 176.9 | 969 → 1519 | `q8gpu-smollm2-135m-cpu-q8-20260619.json` |

### Two honest findings

1. **Q8 weight-narrowing buys ~1.49× decode on the GPU** (24.6 vs 16.5 tok/s) and ~25–30%
   on prefill at every length — same forward, same device, only the weight dtype changes.
   The decode path is memory-bound, so cutting weight traffic ~4× directly raises throughput.
2. **On 135M the CPU wins outright** — cpu-q8 decode 176.9 tok/s is **7.2×** the GPU, and CPU
   prefill (batched GEMM) is **40–75×** the GPU's single-token-looped device prefill. The GPU
   path is **launch-bound** (~330 device ops/token × a fixed dispatch tax that dwarfs 135M's
   per-op compute), the same regime the CUDA/RTX-4070 lane documents — now confirmed on a
   second vendor. The device path is the architecture that scales to models too big for CPU
   residency, **not** a win at 135M. Lever: batched device prefill + capture-replay graph
   (`Async`/`GraphCompile` both `false` in the RX 7600 caps today).

### Verification

- Correctness gated on the real GPU: `TestHALVulkanQ8ForwardMatchesComputeQ8` →
  **prefill cosine = 1.0, step cosine = 1.0**; `TestHALVulkanForwardMatchesNative` →
  argmax-exact, cosine 1.0. The throughput win is reuse + narrower traffic, not a numerics
  shortcut.
- Each row read directly from its committed JSON (`decode.tok_per_sec`, `prefill[].tok_per_sec`).
- `precision`/`backend.selected`/`backend.tier` fields in each artifact make the provenance
  self-describing (e.g. gpu-q8: `precision=Q8_0`, `selected=vulkan`, `tier=discrete:AMD Radeon RX 7600`).

---

## GPU/CPU Q8 Crossover — the device path catches the CPU as the model grows (2026-06-19)

**Date:** 2026-06-19
**Commit:** `7bf666b` (unblocked by the `8c74fd9` q8_matmul input-tiling fix)
**Files:** `fak/experiments/gpu/crossover-qwen2.5-1.5b-{gpu,cpu}-q8-20260619.json`
**Doc:** `fak/experiments/gpu/CROSSOVER-1P5B-RX7600-20260619.md`

### What This Adds

The 135M GPU result above showed the device path **launch-bound** — 7.2× behind the CPU. The
obvious question: does that gap close on a bigger model, where the per-token GEMM is large
enough to amortize the fixed ~330-op/token dispatch tax? Measured on Qwen2.5-1.5B Q8 (the
`q8_matmul` shader's old inDim≤2048 cap, which the 1.5B FFN's inDim=8960 exceeded, was lifted
in `8c74fd9` — verified bit-correct by `TestVulkanQ8MatMulWideInput`, cosine ≥ 0.9999).

### Results — Q8 decode tok/s, GPU (Vulkan RX 7600) vs CPU (pure-Go legacy)

| model | CPU Q8 decode | GPU Q8 decode | **CPU / GPU ratio** |
|---|---:|---:|---:|
| SmolLM2-135M | 176.9 | 24.6 | **7.2×** |
| Qwen2.5-1.5B | 18.4 | 15.9 | **1.16×** |

The CPU's lead collapses **7.2× → 1.16×** as per-token compute grows ~11×. This is direct
evidence for the device-path thesis: the GPU wins as the model grows (one more size step, 3B+,
likely flips it to a GPU win). The launch-bound regime is a small-model artifact, not a
ceiling.

### Honest fences

- **Decode only.** Prefill still favors the CPU heavily (the device prefill loops single tokens
  — HAL prefill isn't batched — so it runs at decode speed; the CPU batches its prefill GEMM).
  Batched device prefill is the standing next lever; it does not affect the decode crossover.
- A transient large-prefill-shape VRAM-allocation panic exists on the 1.5B (the pool's
  drain-and-retry usually absorbs it; a smaller `-prefill-sizes` avoids it). The decode number
  is stable across reps.

### Verification

- Each ratio read from the committed JSON `decode.tok_per_sec` (GPU 15.900, CPU 18.428 → 1.16×;
  135M GPU 24.620, CPU 176.898 → 7.19×).
- Q8 device GEMM bit-close to the CPU Q8 reference at the 1.5B FFN width
  (`TestVulkanQ8MatMulWideInput`, in=8960, cosine ≥ 0.9999); HAL forward gate argmax-exact.

---

## Session Value-Stack Results (SmolLM2-135M Q8)

**File:** `fak/SESSION-VALUE-STACK-RESULTS.md`

### What This Measures

Compares three arms running the **same Q8 forward pass**:
- **A — naive-stateless**: Re-prefills entire context every turn (common local pattern)
- **B — per-agent-KV**: Prompt-cache/persistent KV per agent, no cross-agent sharing
- **C — fak fused**: Prefix prefilled once + cloned into C agents, batched decode

### Results

| Turns | Agents | Prefix | Naive (A) | Tuned (B) | fak (C) | A/C | B/C |
|---|---|---|---|---|---|---|---|
| 8 | 4 | 512 | 135.1s | 32.4s | 12.0s | **11.2×** | 2.70× |
| 16 | 4 | 512 | 409.4s | 67.9s | 28.2s | **14.5×** | 2.41× |

### Key Point

The **11.2–14.5×** value-add is **vs naive stateless serving**, not vs SGLang or any other tuned baseline. This is the "common local pattern" comparison.

> **Low-T anchor for the high-T ladder above.** This T=8/16 authority-shape result
> (C=4, D=24) is the conservative point; the 2026-06-19 high-T ladder pushes the same
> A-vs-C comparison to T=512 → 139.3× by isolating T-scaling with a smaller per-turn
> step. Both are vs the same naive-stateless baseline; they differ only in shape.

---

## Baseline Comparisons: What Each Number Means

| Number | Compares Against | Regime |
|---|---|---|
| 4.58× → 6.95× | Full re-prefill per request | RadixAttention live ladder (135M → 1.5B), climbing to the 7.50× ceiling |
| 7.50× | Token count reduction | Theoretical compute saved (deterministic, model-independent) |
| 86.7% | SGLang's published 50-99% band | Cache hit rate (FCFS 62.1% → cache-aware, 100% of optimal) |
| 11.2–14.5× (T=8/16) → 139.3× (T=512) | Naive stateless (no KV persistence) | Session value-add, O(T²)→O(T) as T grows |
| 2.4–2.7× | Tuned single-tenant (per-agent KV) | Marginal value over warm cache |

---

## Cross-Index

### SGLang RadixAttention Paper
- **Source:** Lianmin Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," arXiv:2312.07104; NeurIPS 2024
- **fak replication:** `fak/RADIXATTENTION-RESULTS.md`
- **Claim:** fak achieves 86.7% hit rate (inside SGLang's 50-99% band)

### SmolLM2-135M Reference
- **Role:** In-kernel bit-exact anchor for GPU/CPU equivalence gates
- **Proof:** `fak/IN-KERNEL-MODEL-DESIGN.md` R0–R14
- **Status:** Proven bit-for-bit vs HF oracle

---

## Reproduce

```bash
# RadixAttention benchmark
go run ./cmd/radixbench \
  -dir internal/model/.cache/smollm2-135m \
  -quant \
  -out experiments/radixattention/radixbench-smollm2-135m-q8.json

# Session value-stack
go run ./cmd/sessionbench \
  -turns 8,16,32 -agents 4 -prefix 512 -decode 24 -result 48 \
  -out experiments/session/smoke-smollm2.json
```

---

## Tombstoned/Outdated Claims

The following claims have been superseded or should not be used:

| Old Claim | Status | Replacement |
|---|---|---|
| "~13× speedup, P=512,T=5,C=5" | ❌ Not found in committed evidence | Use 4.87× (RadixAttention) or 11.2–14.5× (value-stack) |
| "SmolLM2-135M achieves ~370s → ~30s" | ❌ No committed artifact for this exact config | See committed results above |
| Any uncommitted/transient benchmark numbers | ❌ Must ship via commit + JSON | See authority table |

---

## Next Model Results Template

When benchmarking a new model, add an entry following this structure:

```markdown
### Model-Name Results

**Date:** YYYY-MM-DD
**Commit:** <hash>
**File:** `path/to/artifact.json`

| Metric | Baseline | Optimized | Speedup |
|---|---|---|---|
| Wall-clock | XXX ms | YYY ms | **Z.Z×** |
```

---

## DOS Verification Discipline

Every claim in this document is backed by:
1. **Committed artifact** (JSON in repo)
2. **Git commit** with `dos_commit_audit` verification
3. **Reproducible command** in "Reproduce" section

No claim exists without a traceable source.
