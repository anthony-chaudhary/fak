# SESSION-VALUE-STACK-RESULTS — the net value-add of the fused agent kernel on a realistic multi-agent session, measured on Apple M3 Pro

> **📊 AUTHORITY:** This document's session value-stack results (11.2–14.5× vs naive) are centrally
> indexed in **[BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md)** alongside all other benchmark
> claims. That document is the single source of truth with full traceability.

> **What this answers.** The single-stream and batched head-to-heads
> (`MODEL-BASELINE-RESULTS.md`, `M3-LLAMACPP-RESULTS.md`, `LLAMACPP-HEADTOHEAD-RESULTS.md`)
> already settled the *throughput* question honestly: fak's pure-Go forward pass is **in the
> race** with llama.cpp single-stream (≈0.46× decode / 0.15× prefill on M3 Q8_0) but **not
> faster** — the moat was never raw tok/s. This document measures the axis that *is* the
> point: on a **realistic long, multi-agent session** (the "50+ turns, 5+ agents" regime),
> how much total work does the fused kernel's **reuse** save versus what a naive per-turn
> setup actually pays? The answer decomposes cleanly, and the headline number is scoped to
> the baseline it is measured against — no "40× faster than llama.cpp" sleight of hand.
>
> Box: Apple **M3 Pro** (6P+6E, 36 GB unified). Model: **Qwen2.5-1.5B-Instruct** (the
> realistic model fak runs in pure Go here) + **SmolLM2-135M** (the fast model for the full
> scaling sweep). Precision Q8_0. Native `go run` (macOS, not the WDAC-blocked Windows host).
> Harness: `cmd/sessionbench`. Raw JSON under `experiments/session/`.

## The workload (what a 50-turn, 5-agent session is)

`C` concurrent agents share one long **prefix** (system prompt + tool schemas, `P` tokens).
Each runs `T` turns; a turn decodes `D` assistant tokens, then ingests `R` private
tool-result tokens between turns. So each agent's context grows `P → P + T·(D+R)`: a big
shared preamble, short answers, and a per-agent context that grows every turn. This is the
direct surface the agent-serving claim needs.

## The three arms (same model, same bit-identical kernels — the delta is PURE work-reuse)

Every arm runs the **same** Q8 forward pass. The f32 batched-decode and prefix-clone paths
are **proven bit-for-bit identical** to serial `Step` (`TestBatchedDecodeMatchesSerial`,
`TestBatchFromPrefixMatchesIndependentPrefill`, both green native on this M3), so the arms
differ **only** in which work is reused vs redone — never in the numbers they produce. The
value-add is not bought by cutting a correctness corner.

| arm | what it models | prefix prefill | per-turn prefill | decode |
|---|---|---|---|---|
| **A — naive-stateless** | the common local pattern: a stateless API / `llama-cli -p <full prompt>` each turn | re-prefills the **whole context** every turn (O(T²), the prefill's own attention is O(L²)) | — | serial |
| **B — per-agent-KV** | a careful single-tenant setup: prompt-cache / persistent KV per agent, no cross-agent sharing, no batching | once **per agent** (C×P) | incremental (R only) | serial |
| **C — fak fused** | the kernel: prefix prefilled **once** + cloned into C agents, batched decode, incremental ingestion | **once** total | incremental (R only) | **batched** (one weight stream serves all C) |

**Net value-add = A/C** (vs the naive pattern) and **B/C** (the honest vs-tuned-single-tenant
number). They decompose: **A→B is the turn-tax** (KV persistence vs re-prefilling every turn);
**B→C is prefix reuse** (P once, not ×C) **+ decode batching** (one weight stream for C agents).

## Methodology — live where tractable, measured-rate where not (never modeled)

Arms **B and C are run end-to-end LIVE**: every prefill and decode is a real kernel call, so
attention's growth with context length is captured **exactly** (the per-turn cost rises
through the session, as it must). Arm **A's re-prefill is O(T²) and intractable to run at
50×5** (~hours on a 1.5B CPU forward pass), so it is computed from `prefillCost(L)` — prefill
wall-clock **measured at sampled lengths** spanning the session, which captures the prefill's
own **O(L²) attention term** (not assumed linear) — summed over the **exact** per-turn context
lengths, ×C. Arm A's *decode* is byte-identical serial work to arm B's, so `A_decode :=` arm
B's measured live decode. A `-validate` pass runs arm A **fully live** at a small scale and
confirms the computed arm-A wall-clock matches.

**Conservative by construction:** fak's incremental prefill is charged live (no discount), the
prefill model is measured (the quadratic is real — see below), and the `B/C` number is reported
right next to `A/C` so the headline is never read out of its baseline.

### Evidence the prefill quadratic is real (why arm A is expensive, measured not assumed)

`prefillCost(L)` is sampled live; on SmolLM2 the throughput **falls** as length grows because
prefill attention is O(L²) — a linear model would badly under-price arm A's long-context
re-prefills:

| prefill length L | wall-clock | tok/s | linear prediction from L=256 |
|---:|---:|---:|---:|
| 256 | 0.65 s | 396 | — |
| 512 | 2.12 s | 241 | 1.29 s |
| 896 | 4.27 s | 210 | 2.26 s |
| 1536 | 9.24 s | 166 | 3.88 s |
| 2176 | 13.90 s | 157 | 5.49 s |
| 2816 | 20.38 s | 138 | 7.11 s (actual is **2.9×** this) |

The harness sums the **measured** curve over arm A's exact per-turn contexts, so its
re-prefill cost is grounded, not extrapolated from a single rate.

## Results — the scaling sweep (SmolLM2-135M, full live B/C, P=512, D=24, R=48)

The value-add **grows with session length**, exactly as the O(T²) re-prefill predicts:

| T (turns) | C (agents) | A naive | B tuned | C fak | **net vs naive (A/C)** | vs tuned (B/C) | turn-tax (A/B) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 4 | 135.1 s | 32.4 s | 12.0 s | **11.2×** | 2.70× | 4.2× |
| 16 | 4 | 409.4 s | 67.9 s | 28.2 s | **14.5×** | 2.41× | 6.0× |
| 32 | 4 | <!--T32A--> | <!--T32B--> | <!--T32C--> | **…×** | …× | …× |

The vs-tuned (B/C) number holds steady at **~2.4–2.7×** (prefix-reuse + decode batching — the
genuine marginal value over an already-warm KV cache), while the turn-tax (A/B) **grows with
T** as naive's O(T²) re-prefill compounds. The headline below uses a **4× larger prefix
(P=2048)**, which raises both: in arm A every agent re-prefills the *full* prefix every turn
(prefix cost ~C·T·P), while fak pays it once.

## Results — the headline (Qwen2.5-1.5B-Instruct, the realistic model, 50 turns × 5 agents)

> **Pending live run.** The `…` cells below fill from a `sessionbench` Qwen2.5-1.5B
> 50×5 run on a bench node — see `PLAN-model-ladder-qwen36-2026-06-19.md`.
> Until that artifact lands, this table documents the measured *shape*, not the numbers.

| metric | value |
|---|---|
| session | T=50 turns, C=5 agents, P=2048 prefix, D=32, R=64 |
| arm A (naive-stateless) | … |
| arm B (per-agent-KV, tuned single-tenant) | … |
| arm C (fak fused) | … |
| **net value-add vs naive (A/C)** | **…×** |
| net value-add vs tuned single-tenant (B/C) | …× |
| turn-tax (A/B) | …× |

### Methodology validation (computed arm A vs arm A run fully live)

> **Pending live run.** The arm-A-computed-vs-fully-live validation row fills from the same run.

## The honest scoping (read before quoting the headline)

- **The big multiple is vs the *naive* baseline** — re-prefilling the whole context every
  turn, one process per agent, no batching. This is **not** a strawman: it is exactly what a
  stateless-API call or `llama-cli -p <full prompt>` per turn does, and it is the common shape
  for hand-rolled local agent loops. But it must be quoted **as** the naive comparison.
- **Vs a *tuned* single-tenant stack** (prompt-cache / persistent KV, the B arm) the win is
  **~2–3×** — prefix-reuse + decode batching, the genuine marginal value over "already having
  a KV cache." This is consistent with the repo's standing honesty bound (the serving win vs a
  tuned SGLang is ~2–2.5×; `TURN-TAX-RESULTS.md` §3.1).
- **No single lever is novel** — continuous batching, prefix/KV reuse, and prompt caching are
  all established. The contribution is the **integrated kernel** that delivers all of them
  **correctly, in one binary, while preserving per-agent KV ownership** (every agent can
  `Evict`/`Clone` its own span) — the property a shared-slot serving engine structurally
  cannot offer, and the one the safety floor below is built on.
- **fak is not faster than llama.cpp.** A tuned `llama-server` with parallel slots batches and
  reuses KV too; on raw throughput it is ahead (`M3-LLAMACPP-RESULTS.md`). The value-stack here
  is the *reuse-vs-redo* axis within a fixed engine, not a throughput head-to-head. The raw
  single-stream gap (decode ≈0.46×, prefill ≈0.15× on M3 Q8_0) is a separate, honestly-reported
  axis; the prefill half is being closed from the GPU side by the opt-in Metal GEMM lane
  (`internal/metalgemm`, `-tags fakmetal`, default build stays pure-Go) — that work is
  orthogonal to and composes with the reuse stack measured here.

## The moat that does not depend on any of the above — the safety floor (§1)

The efficiency stack is the *ceiling*; the **floor** is the engine-agnostic trust property,
and it is what makes the saved work *real* rather than *wasted*. A 50-turn session that gets
derailed by one poisoned tool result, or that executes one destructive op, is **total waste** —
every turn after the derailment is compute spent producing a wrong or dangerous trajectory. The
kernel's deterministic floor prevents exactly that, live on this box (`fak turntax --suite
turntax-airline`):

| axis | baseline (SOTA loop) | fak | delta |
|---|---:|---:|---|
| injections admitted to context | 1 | **0** | quarantined |
| destructive ops executed | 1 | **0** | denied |

Both are verdicts the model **did not author**, reproducible on **any** backend (a local 1.5B
or a frontier API you do not own), and the happy-path control (`turntax-happy`) saves exactly
**0** — the anti-inflation gate. So the net value-add is two things at once: the **reuse**
above (self-host regime), and the **correctness floor** that keeps a long session from
becoming expensive garbage (every regime). The second is the non-optional one.

## Witnesses (the DOS discipline — every claim closed by a mechanical witness the author did not write)

| claim | witness (re-runnable) | what it rules out |
|---|---|---|
| the three arms produce the **same** tokens (value-add is reuse, not a numerics shortcut) | `go test ./internal/model -run 'TestBatchedDecodeMatchesSerial\|TestBatchFromPrefixMatchesIndependentPrefill'` (green native on this M3) | "fak is faster because it computes something cheaper/wrong" |
| arm A's O(T²) projection is faithful to a live run | the `live_validate` block: arm A run **fully live** at a small scale, `anchored_computed_over_live ≈ 1.0` | "the naive baseline cost is modeled/inflated" |
| the prefill quadratic is measured, not assumed | `prefill_model` samples in the JSON (throughput falls with length) | "arm A's re-prefill is extrapolated from one rate" |
| arm-A decode == arm-B decode (no double-count) | `decode_A_over_B ≈ 1.0` in `live_validate` | "arm A's decode is padded" |
| the value-stack arithmetic is correct | `go test ./cmd/sessionbench` (interpolation, closed-form quadratic, extrapolation) | "the count formula is wrong" |
| the safety floor is real and not inflated | `fak turntax --suite turntax-airline` (1→0, 1→0) + `turntax-happy` (saves 0) | "the floor is a fixed discount / cherry-picked" |
| contention did not bias the headline | `prefill_anchor` per cell re-scales the sampled model to arm B's live prefix prefill | "the number is a busy-box artifact" |

The literal `dos verify` truth-syscall over the ship commit runs on the Windows DOS host
(`C:\work\fleet`, the `dos` package is not installed on this Mac); here the discipline is the
witness set above + the committed JSON artifacts + an independent adversarial skeptic pass.

## Reproduce

```sh
# headline — Qwen2.5-1.5B, 50 turns × 5 agents (realistic model)
FAK_WORKERS=6 go run ./cmd/sessionbench \
  -hf <qwen2.5-1.5b-instruct-snapshot> -lean \
  -turns 50 -agents 5 -prefix 2048 -decode 32 -result 64 \
  -out experiments/session/headline-qwen-50x5.json

# scaling sweep — SmolLM2-135M, full live B/C
go run ./cmd/sessionbench -turns 8,16,32 -agents 4 -prefix 512 -decode 24 -result 48 \
  -out experiments/session/smoke-smollm2.json

# the safety floor (model-free kernel replay)
fak turntax --suite turntax-airline       # 1 injection→0, 1 destructive→0
fak turntax --suite turntax-happy          # control: saves 0

# the bit-identity gates the whole value-stack rests on
go test ./internal/model/ -run 'TestBatchedDecodeMatchesSerial|TestBatchFromPrefixMatchesIndependentPrefill' -count=1
```

## Bottom line

> **Pending live run.** A one-paragraph honest summary lands here once the headline run
> completes, quoting both scoped numbers (A/C vs the stateless arm, B/C vs tuned single-tenant).
