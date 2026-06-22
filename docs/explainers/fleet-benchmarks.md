---
title: "The fak Fleet Benchmark Suite — Run the Five Headline Demos Yourself"
description: "Five model-agnostic fleet benchmarks you can reproduce in minutes with `go run` — no GPU, no model weights, no API key: fan-out to 1024 sub-agents, a 50×50 turn-tax sweep, the turn-tax A/B + safety floor, RadixAttention cache hit rate, and context-changing token accounting. Every number traces to BENCHMARK-AUTHORITY."
---

# The fak Fleet Benchmark Suite — explore it yourself

> **What this page is.** A single place to *run* the five benchmarks that show what
> `fak` buys a **fleet** of agents — cross-agent cache reuse, turn-tax elimination, and
> fan-out — and to read each one honestly. All five are **model-agnostic kernel demos**:
> they drive the real `fak` kernel (`k.Syscall`, the process-global vDSO cache, the
> ctx-MMU, `NewBatchFromPrefix`) but need **no model weights, no GPU, and no API key**, so
> the headline numbers reproduce on any laptop in minutes. Every figure here traces to
> **[BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md)** (the single source of truth)
> and the on-box witnessed run in
> [`GLM52-PURE-KERNEL-AND-AGENT-TURN-DEMOS-RESULTS-2026-06-21.md`](../notes/GLM52-PURE-KERNEL-AND-AGENT-TURN-DEMOS-RESULTS-2026-06-21.md) §3.
>
> If you want the *why* behind the win first, read
> [KV cache for agentic context](kv-cache-agentic-context.md) and
> [SOTA optimizations fak sits on top of](sota-optimizations.md). If you want to *watch*
> them in a browser instead of running them, jump to [Watch them live](#watch-them-live).

---

## TL;DR — the five headline numbers

Each row is one command you can paste from the repo root. The **honest baseline** column
is the one that matters: the eye-catching multiples are mostly against a *naive / cold*
reference, while the real fak-only win is the **cross-agent** reuse on top of an
already-warm per-agent cache (the baseline a tuned vLLM / SGLang / provider stack gives
you). Both are shown — never just the flattering one.

| Demo | Headline result | The honest baseline (read this) |
|---|---|---|
| **`fanbench` fan-out, N=1024** | **1,005** sibling-only tool-result saves · **61.7%** of the multi-agent token tax clawed back · **72.8×** parallel critical-path speedup | The cross-agent dedup is measured *on top of* a warm per-agent prompt cache. The `61.7%` is modeled cache economics (saturates there); the `72.8×` is latency that saturates past N≈256 as the fold dominates. |
| **`fleetbench` 50×50 corner** | deletes **2,344 / 2,500** tool calls · **+370** cross-agent turns over isolated worlds | The `+370` cross-uplift is the real fleet-only win (a measured tier-2 vDSO path-swap); it is **read-fleet only** — even a ~1% write rate flips it negative under the coarse eraser. |
| **`fak turntax` airline / happy** | **9** turns saved on airline (forced 5 + elision 4) · **0** on the clean control · safety floor: injections **1→0**, destructive **1→0** | The 9 is a *cache-favorable slice* (~64% addressable), **not** the ~0.7% real-world rate; turn-savings are **self-host-only**. The **safety floor** is the moat — engine-agnostic, on a separate axis. |
| **`radixbench`** | **77–88%** cache hit across workloads · agents: FCFS **62.1% → 86.7%** cache-aware · policy-evict freed 8 tokens, kept the sibling | Hit rate is hardware-/model-independent, so fak-on-CPU vs SGLang-on-GPU is a fair axis; `86.7%` is inside SGLang's published 50–99% band. The token-speedup is vs a *cold* baseline. |
| **`ctxdemo` fleet-5×50** | **1.26M** cold tokens → **35,495** with fak (**35.5×** vs cold) | The 35.5× is vs the cold no-cache *reference*. Against the honest serving baseline (warm per-agent KV) the win is **1.1×** — both printed side by side. |

> **One framing law for the whole suite.** Compare against the **best already-shipped
> baseline**, state the absolute number, and mark every `naive`/`cold` multiple so it can
> never read as a SOTA win. `fak` does **not** beat vLLM / SGLang / llama.cpp on raw
> tokens-per-second and never claims to — see
> [one binary is the whole surface](one-binary-one-surface.md). What these benchmarks
> measure is the orthogonal axis: *how much redundant work a fleet's kernel can delete,
> exactly and safely.*

---

## Before you run anything

**Prerequisites — that's the whole list:**

- **Go 1.26+** and a clone of the repo. Run every command **from the repository root**
  (the Go module *is* the repo root).
- **No model weights, no GPU, no API key** for any of the five. They replay class-labeled
  traces and synthetic workloads through the real kernel and read the kernel's *own*
  counters; the headline numbers are deterministic and seeded, so a fixed
  `(profile, grid, trials, seed)` reproduces the identical surface byte-for-byte.
- Artifacts (`fanout.{json,csv}`, `fleet-sweep.{json,csv}`, `turntax-report.json`,
  `radix.json`) are written to the working directory and are regenerable — nothing to
  commit.

> These are the **kernel** demos. They are distinct from the **model-ladder** benchmarks
> (`sessionbench`, `modelbench`, the live `radixbench -hf …` arm), which *do* need a real
> checkpoint on disk to produce wall-clock tok/s — those are indexed in
> [BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md). This page stays on the
> model-agnostic floor so anyone can reproduce it.

**Two honesty axes, kept strictly apart** (the same discipline runs through every demo):

1. **Measured on the real kernel** — cross-agent dedup, turn-tax levers, hit rate, and the
   safety floor are *kernel events* the model did not author (`VDSOHits`, `Transforms`,
   `Quarantines`, `Denies`), proven by a real ON/OFF path-swap or a shared-vs-isolated
   ablation, with an **exactly-zero** anti-inflation control.
2. **Modeled by a transparent cost model** — only the *price* of a turn (tokens, dollars,
   latency) and the prefix-cache economics are modeled, with every knob exposed as a flag.
   The two halves are never blended.

---

## 1. `fanbench` — one master goal → N sub-agents (the fan-out topology)

**What it measures.** The orchestrator-worker pattern (one lead decomposes a goal, spawns
N sub-agents, folds their results), swept from N=1 to **N=1024** — the regime no public
benchmark maps. It prices the cross-agent tool-result dedup the fan-out structure deletes,
plus the exact `(N−1)·prefix` prefill the kernel never redoes because `NewBatchFromPrefix`
prefills the shared master-goal prefix once and clones it bit-identically into all N
sub-agents.

**Why it matters in a fleet.** When N agents decompose one goal, they read the same shared
sources. A naive framework re-ships the full system+goal prompt per sub-agent; fak does the
shared prefill once for the whole wave.

**Run it:**

```bash
go run ./cmd/fanbench -agent-max 1024 -grid log
```

**The N-ladder corner** (research profile, the headline surface):

| N | calls | shared | isolated (warm) | **cross** | tax clawed back | parallel speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1,028 | 785 | 536 | 255 | 61.7% | 57.7× |
| 512 | 2,052 | 1,569 | 1,069 | 483 | 61.7% | 66.9× |
| **1024** | **4,100** | **3,155** | **2,152** | **1,005** | **61.7%** | **72.8×** |

At N=1024 the interleaved fan-out deletes **3,155 of 4,100 calls (77%)**, of which **+1,005
is the cross-agent bonus** the same sub-agents run solo could not get.

**Honest fences.**
- `cross_uplift` is a **fak-vs-fak** SHARED-vs-ISOLATED ablation (the fan-out's win over
  running the sub-agents apart), **not** a head-to-head over a tuned shared-prefix engine —
  SGLang/RadixAttention and vLLM prefix caching occupy the same prefix lever.
- The `61.7%` tax-clawed-back is **modeled** prompt-cache economics (Anthropic-style
  read 0.1× / write 1.25×); it saturates at the `1 − 0.9P/(P+S+D+fold) ≈ 0.618` asymptote.
- Fanning out to **N=1 is a net loss** (the orchestration fold costs more than doing the
  goal yourself) — surfaced honestly, not hidden.
- This is a **latency / kernel-cost** axis, **not** task quality (no ground-truth
  sub-results; coverage@N is tracked separately).

**Deep dive:** [`docs/benchmarks/FANOUT-BENCH-RESULTS.md`](../benchmarks/FANOUT-BENCH-RESULTS.md).

---

## 2. `fleetbench` — the 2-D turn-tax surface (turns × agents)

**What it measures.** `fak turntax` prices one agent; this sweeps the full **1..50 × 1..50**
grid — 2,500 cells — of A independent agents that happen to overlap. The kernel's tier-2
vDSO cache is keyed `(tool, args-sha256, world-version)` and is **process-global**, so when
A agents read the same reference data the first pays a cold round-trip and every other
agent's identical read is a tier-2 hit the kernel counts itself (`Counters.VDSOHits`). Each
cell is ablated **shared-world fleet vs per-agent-isolated worlds**.

**Why it matters in a fleet.** A research / monitoring / support-lookup fleet mostly reads
shared reference data. The cross-agent uplift is the turns *sharing* buys that A
independent agents cannot get — and it is **linear in agent count** but saturating in turns.

**Run it** (the 50×50 read-heavy corner, as witnessed):

```bash
go run ./cmd/fleetbench -agents 50 -turns 50 -trials 24 -profile read-heavy -granularity resource
```

```
T=50 A=50  calls=2500  shared=2344  isolated=1974(warm)  cross=370
tokens_saved_shared=3,094,080   $12.66 saved (shared)
```

The read-fleet corner **deletes 2,344 / 2,500 calls (94%)** with **+370 cross-agent turns**
over isolated (warm per-agent KV) worlds. Run it without `-agents/-turns` to sweep the full
2,500-cell heatmap.

**Honest fences.**
- The `+370` is **measured** (the kernel's own VDSOHits via a shared-vs-isolated path-swap);
  the **no-share** control is **exactly 0 across all 2,500 cells**, so a positive number is
  never the benchmark flattering itself.
- A `cross_uplift` of +370 is 370 **tool round-trips** served from a peer's cached result —
  *not* 370 saved model *reasoning* turns.
- **Read-fleet only.** Under the coarse v0.1 (`global`) eraser a **~1% write rate flips
  sharing from a big win to a net loss**, because one write bumps the whole world version.
  The finer **`resource`** eraser keeps 97% of the no-write uplift even at a 1% write rate —
  hence `-granularity resource` above. Sweep `-granularity global|namespace|resource` to see
  the crossover move.
- Unlike the in-tensor KV story, this is **harness-level result caching**, so it is
  available to an API consumer who fronts a read-heavy fleet with **one fak gateway**.

**Deep dive:** [`docs/benchmarks/FLEET-SWEEP-RESULTS.md`](../benchmarks/FLEET-SWEEP-RESULTS.md).

---

## 3. `fak turntax` — the turn-tax A/B and the safety floor

**What it measures.** Two distinct things, kept structurally apart:

1. **The safety floor (the moat).** On a 14-call airline-support slice, the kernel quarantines
   a poisoned tool result out of context (`Quarantines`) and refuses a destructive
   `delete_account` (`Denies`) — a deterministic completion/integrity delta the model did not
   author, reproducible **on any backend including a frontier API you do not own**.
2. **The efficiency upside (self-host only).** When a SOTA tool-calling loop hits an error
   code, malformed args, or a duplicate read, the documented recovery is to re-prompt — an
   extra turn. fak's 1-shot path resolves the same condition *inside the syscall*.

**Why it matters in a fleet.** The safety floor is the non-optional reason to run the kernel
at all, and it scales to every agent regardless of which engine answers the call.

**Run it** (the demonstration slice, then the anti-inflation control):

```bash
go run ./cmd/fak turntax --suite turntax-airline
go run ./cmd/fak turntax --suite turntax-happy
```

| Suite | turns saved | breakdown | vDSO ON / OFF | safety floor (separate axis) |
|---|---:|---|---|---|
| `turntax-airline` | **9** | forced 5 (grammar + dedup) + elision 4 (pure + static) | 9 / 2 → vDSO = **7 turns** | injections admitted 1 → fak **0**; destructive executed 1 → fak **0** |
| `turntax-happy` | **0** | — the clean-path control: it inflates nothing | 0 / 0 | base 0 / fak 0 |

The vDSO contribution (7) is proven by a **real ON/OFF path swap** (`SetVDSO(false)` drops
the win to grammar-only 2), and it equals the live `Counters.VDSOHits` — not arithmetic.

**Honest fences.**
- **The 9 is a cache-favorable slice** (~64% of calls addressable), built so every lever
  fires once. On real tau2-airline the addressable vDSO rate is **~0.7%** — so do **not**
  extrapolate "agents save 9 turns." The `turntax-happy` control saves exactly **0**, by
  construction and by test.
- The efficiency win is **self-host / provider-ships regime only**; an **API consumer gets
  the safety floor and none of the turn-savings.** No single lever is novel (grammar repair,
  TVCache-style dedup, prompt caching are all established) — the only novelty is the
  in-syscall assembly. The right serving baseline is **~2–2.5× vs tuned SGLang**, not 5–15×.
- The safety floor is reported on a **deliberately separate axis** and never folded into the
  turn count. (Add `--breakeven` to price the ~0.7% real rate: 0.33 turns/session.)

**Deep dive:** [`docs/benchmarks/TURN-TAX-RESULTS.md`](../benchmarks/TURN-TAX-RESULTS.md).

---

## 4. `radixbench` — RadixAttention prefix reuse + cache-aware scheduling

**What it measures.** fak's KV-cache prefix reuse against SGLang's **RadixAttention**
(arXiv:2312.07104 / NeurIPS 2024) on the metric SGLang's own paper headlines: **cache hit
rate** — the fraction of prompt tokens served from cache instead of recomputed. That metric
is **hardware- and model-independent** (a function of *workload × matching algorithm* only),
so fak-on-CPU vs SGLang-on-GPU is a *fair* head-to-head on this axis. fak runs the same
algorithm (radix tree + longest-prefix match + LRU-leaf eviction, `internal/radixkv`).

**Why it matters in a fleet.** Cache-aware scheduling recovers hit rate a naive FCFS order
thrashes away — exactly the fleet-scheduling lever that turns shared prefixes into saved work.

**Run it** (synthetic workloads, no model needed):

```bash
go run ./cmd/radixbench -scale 1
```

| Workload | reqs | cache hit | cross-subtree reuse | bounded sched (FCFS → cache-aware) |
|---|---:|---:|---:|---|
| few-shot | 16 | 88.2% | 1.00× | 88.2% → 88.2% (100% of optimal) |
| multi-turn-chat | 8 | 79.5% | 2.50× | 79.5% → 79.5% |
| tree-of-thought | 27 | 77.2% | 1.40× | 77.2% → 77.2% |
| **agents (5×6)** | 30 | 86.7% | 1.48× | **62.1% → 86.7%** (cache-aware lift) |

The agents hit rate of **86.7%** is inside SGLang's published 50–99% band; the cache-aware
scheduler lifts FCFS's **62.1% → 86.7%** (100% of the DFS-optimal bound the paper proves).

**Honest fences.**
- The **policy-eviction witness** is the one capability an opportunistic LRU radix cache
  structurally cannot offer: a verdict evicts a *named* (e.g. poisoned) prefix — here it
  freed exactly **8 tokens and kept the benign sibling warm** — eviction by governance, not
  memory pressure. Same primitive as SGLang, opposite control.
- The prefill-token speedup radixbench also prints is measured vs a **cold no-cache**
  baseline — a worst-case reference, not a serving baseline anyone ships.
- The deterministic hit rates reproduce bit-for-bit across platforms (Windows x86_64 vs Mac
  M3 arm64); add `-hf <snapshot> -lean` for the live wall-clock arm on a real checkpoint.

**Deep dive:** [`docs/benchmarks/RADIXATTENTION-RESULTS.md`](../benchmarks/RADIXATTENTION-RESULTS.md)
· authority: [RadixAttention model ladder](../../BENCHMARK-AUTHORITY.md).

---

## 5. `ctxdemo` — context-changing fleet token accounting

**What it measures.** The exact, **timing-free** prefill-token work each strategy performs
in the multi-agent, multi-turn, long-context regime — the one where the context *changes*
every turn as tool calls land heterogeneous, variable-sized results. Decode is excluded
(it's generated, not re-read), so this is a load-independent, hardware-independent floor.

**Why it matters in a fleet.** It puts the three strategies side by side in one number per
scenario: cold re-prefill (naive), warm per-agent KV (the honest serving baseline), and fak
(cross-agent prefix sharing on top of the warm cache).

**Run it** (instant, no model, CI-usable):

```bash
go run ./cmd/ctxdemo -print
```

```
scenario       C   T    P   no-cache    warmKV     fak    fak-win  (ref×)  maxCtx
fleet-5x50     5  50 1024  1,259,857    39,591   35,495    1.1×    35.5×    9569
deep-research  4   5 1536     40,188     9,358    4,750    2.0×     8.5×    2642
```

The 5-agent × 50-turn fleet re-reads **1.26M tokens cold**; fak does **35,495** — **35.5×
vs cold**, and **1.1× on top of an already-warm per-agent KV cache**.

**Honest fences.**
- The **35.5× is vs the cold no-cache reference** (`(ref×)`), a labeled worst-case — not a
  serving baseline. The number that survives contact with a tuned stack is **`fak-win`
  = 1.1×** (vs warm per-agent KV), printed in the same table by design. `deep-research`,
  with a heavier shared prefix and fewer turns, shows a larger **2.0×** honest win.
- This is the *prefill-token floor*, not a wall-clock. For the live race through a real
  in-kernel model, drop `-print` and serve the page (below), or use `-race deep-research`.

**Deep dive:** the command's own header (`cmd/ctxdemo/main.go`) and
[`docs/benchmarking/README.md`](../benchmarking/README.md).

---

## Watch them live

If you'd rather see these drive the kernel in a browser than run them locally, the
[**live demos page**](../demos.html) hosts three of them on a single GCP VM (NVIDIA L4):
the turn-tax race (`turntaxdemo`), the multi-agent context-reuse proof (`ctxdemo`), and a
live model reuse race (`demorace`) — each driving the *real* kernel, not a recording. Run
any of them locally instead:

```bash
go run ./cmd/turntaxdemo   # http://127.0.0.1:8150 — turn-tax race, no model
go run ./cmd/ctxdemo       # http://127.0.0.1:8153 — context reuse (live model if one is on disk)
go run ./cmd/demorace      # the reuse race + the reuse curve
```

---

## The honesty discipline (one place)

Every number on this page obeys the same rules, enforced in CI and in the per-demo tests:

- **The baseline is a warm cache, not a straw man.** The fak-only win is the *cross-agent*
  reuse on top of an already-warm per-agent KV cache (what a tuned vLLM / SGLang / provider
  stack gives you). The big multiples (`35.5×`, `72.8×`, the cold-baseline token speedups)
  are vs a *naive / cold* reference and are always labeled as such.
- **Measured vs modeled is never blended.** Kernel events (dedup, hit rate, turn levers,
  the safety floor) are measured path-swaps with zero-valued anti-inflation controls; only
  the per-turn *price* and the prompt-cache *economics* are modeled, with every knob a flag.
- **`fak` does not race tokens-per-second.** vLLM / SGLang / llama.cpp win raw throughput
  and front-of-prompt prefix reuse; `fak` owns the governance + reuse-exactness band. See
  [one binary is the whole surface](one-binary-one-surface.md).
- **The safety floor is engine-agnostic and on its own axis** — it never inflates an
  efficiency number, and it holds on a frontier API you do not own.
- **No single lever is novel** (a 29-claim prior-art audit scored 0/29 novel); the
  contribution is the *assembly* at the syscall boundary.

---

## Where to go deeper

- **[BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md)** — the single source of truth;
  every number traces to a commit + artifact.
- **[BENCHMARK-GOVERNANCE.md](../../BENCHMARK-GOVERNANCE.md)** — the DOS-centric process
  that creates, verifies, and publishes a claim before it can appear here.
- **[BENCHMARK-GALLERY.md](../../BENCHMARK-GALLERY.md)** — the four generated hero visuals
  (model-card style), each from one source-of-truth JSON with a `--check` CI drift gate.
- **[Benchmarking index](../benchmarking/README.md)** — how to read the baselines, the
  measured-vs-modeled split, and the full tool inventory.
- **[GLM52 witnessed run §3](../notes/GLM52-PURE-KERNEL-AND-AGENT-TURN-DEMOS-RESULTS-2026-06-21.md)**
  — the on-box reproduction these five numbers were taken from, closed by `go test` exit
  codes and benchmark output fields, not self-report.

*Last updated: 2026-06-21*
