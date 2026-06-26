---
title: "fak fan-out benchmark: one goal to N sub-agents"
description: "Prices one master goal decomposed into N sub-agents swept from 1 to 1024, measuring cross-agent dedup and shared-prefix KV reuse on the real kernel."
---

# FANOUT-BENCH-RESULTS — one master goal → N sub-agents, swept N=1…1024 on the real kernel

> **What this measures.** `fak turntax` prices ONE agent; `fleetbench`
> (`FLEET-SWEEP-RESULTS.md`) prices **A independent agents** that happen to overlap.
> This sweep prices the topology neither captures and that **no public benchmark
> maps** (see `experiments/fanout/RESEARCH-BRIEF-fanout-2026-06-17.md`): a **single
> master goal decomposed into N sub-agents** — the orchestrator-worker / lead-subagent
> pattern (Anthropic's lead-researcher → 3–5 subagents, scaled to **N=1024**). The
> regime N≥50 on one goal is unbenchmarked in the literature, and **shared-prefix KV
> reuse across N siblings has never been quantified for it.**
>
> **What is grounded vs modeled (the honest line — same discipline as TURN-TAX §3.2 /
> FLEET-SWEEP).** Two halves, kept strictly apart:
> - **MEASURED on the real kernel.** The cross-agent tool-result dedup is a real
>   `k.Syscall` tier-2 event. Each cell is ablated **SHARED** (the whole fan-out in one
>   world epoch) vs **ISOLATED** (each sub-agent run solo — its own plan + own world),
>   and `cross_uplift = shared − isolated` is the fan-out-ONLY sibling dedup — a measured
>   path-swap, identical in discipline to `fleetbench`. The shared-prefix KV-reuse saving
>   is **exact geometry** — `(N−1)·prefix_tokens` prefill the kernel does not redo because
>   `model.NewBatchFromPrefix` prefills the master-goal prefix ONCE and clones it into all
>   N sub-agents (bit-identical; `model.TestKVPrefixReuseMatchesRecompute` +
>   `cmd/fanbench.TestPrefixReuseFanoutWitness`, wall-clocked by `cmd/fleetserve`).
> - **MODELED by a transparent, knobbed cost model** (`FanoutCostModel`): the token
>   multiplier, the prompt-cache prefix economics, critical-path-vs-total-work latency,
>   throughput, and the saturation knee. Reported separately, never blended into the
>   measured halves.
>
> Reproduce: `wsl bash -c "cd fak && go run ./cmd/fanbench -profile research -trials 16
> -out experiments/fanout/fanbench-research.json -csv experiments/fanout/fanbench-research.csv"`.
> Prefix-scale sweep: add `-prefixes all` for `smoke,small,medium,long,big`; set
> `-model-config /path/to/config.json` or `-model-context N` so `big/max` fills the
> target model context after reserving the private suffix and decode budget.
> Code: `internal/turnbench/fanout.go`, runner `cmd/fanbench/`, tests
> `internal/turnbench/fanout_test.go` + `cmd/fanbench/main_test.go`.

**High-signal external references.** The result should be read against the shared-prefix
serving literature and provider cache economics, not as a claim that FAK invented the
prefix lever: vLLM Automatic Prefix Caching
([docs](https://docs.vllm.ai/en/stable/design/prefix_caching/), [PagedAttention paper](https://arxiv.org/pdf/2309.06180)),
SGLang/RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/), [paper](https://arxiv.org/pdf/2312.07104)),
Hydragen shared-prefix attention ([paper](https://arxiv.org/abs/2402.05099)),
Anthropic prompt caching ([docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)),
OpenAI prompt caching ([docs](https://developers.openai.com/api/docs/guides/prompt-caching)),
and Anthropic's orchestrator-worker research-system writeup
([engineering note](https://www.anthropic.com/engineering/multi-agent-research-system)).

---

## §1 — The headline surface (research-goal profile, 16 seeded trials, sub-turns=4)

`research-goal`: sub-agents mostly read the goal's shared sources (`p_shared=0.55`,
`goal_pool=8`), a 4-read master plan warms the catalog, no writes. Medians over 16 trials
(`experiments/fanout/fanbench-research.csv`):

| N (sub-agents) | calls = plan+N·T | shared_saved | isolated_saved | **cross_uplift** | prefix_tokens_saved | token_mult naive→reuse | **tax_clawed_back** | parallel_speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1    | 8    | 2    | 2    | **0**    | 0         | 1.07→1.26  | **0%** (loss)  | 1.0   |
| 4    | 20   | 10   | 8    | **+2**   | 6,144     | 4.3→2.5    | **54.8%**      | 3.0   |
| 16   | 68   | 46   | 34   | **+12**  | 30,720    | 17.1→7.4   | **60.4%**      | 11.0  |
| 64   | 260  | 191  | 134  | **+58**  | 129,024   | 68.6→27.1  | **61.4%**      | 29.0  |
| 256  | 1028 | 785  | 536  | **+251** | 522,240   | 274→106    | **61.7%**      | 57.7  |
| 1024 | 4100 | 3155 | 2152 | **+1005**| 2,095,104 | 1098→420   | **61.7%**      | 72.8  |

Read four things off this surface:

1. **The fan-out structure deletes turns the same sub-agents run apart cannot.**
   `cross_uplift` (the MEASURED `shared − isolated` path-swap) grows ~linearly with N:
   at N=1024 the interleaved fan-out deletes **3,155 of 4,100 calls (77%)** in one world
   epoch, of which **+1,005 is the cross-agent bonus** the same 1,024 sub-agents run solo
   (2,152 deleted) could not get. `N=1` is **exactly 0** — a lone worker has no sibling to
   share with (the budget-controlled single-agent control; partner of the `no-share`
   profile that is 0 at every N).

2. **The prefix-cache lever claws back ~62% of the multi-agent token tax — and saturates
   there.** `tax_clawed_back` rises from 54.8% (N=4) to a **61.7% plateau** by N≈256. This
   is the modeled prefix-cache economics: the master-goal prefix is written to cache once
   (1.25×) and read cheap by the other N−1 sub-agents (0.1×) instead of materialized N
   times — `1.25 + (N−1)·0.1` vs naive `N·1.0`. The analytic asymptote is
   `1 − 0.9P/(P+S+D+fold) ≈ 0.618`; the measured curve lands on it.

3. **Prefix reuse is `(N−1)·P` prefill the kernel never redoes.** At N=1024 that is
   **2,095,104 prefill tokens** the SHARED arm does not recompute — the exact geometry of
   `NewBatchFromPrefix` (prefill once + N bit-identical clones).

4. **Fanning out to ONE sub-agent is a NET LOSS.** At N=1 `token_mult` is 1.07 (naive) /
   1.26 (reuse) and `net_tokens_saved = −512`: the orchestration fold plus the cache-write
   overhead cost *more* than just doing the goal yourself. This is the literature's "a
   single agent beats trivial multi-agent," surfaced honestly rather than hidden — the
   levers only pay once there are siblings to amortize across.

### The D-001 acceptance grid (N=1 / 100 / 500 / 1000) — host-computed, reproducible (#255)

The §1 surface above uses the geometric N-ladder (the right spacing to read a curve
spanning three orders of magnitude). The D-001 acceptance criteria
([#255](https://github.com/anthony-chaudhary/fak/issues/255)) name the **literal** scale
points **N=100, N=500, N=1000** and ask for **coordination overhead vs a baseline** next to
the cross-agent reuse uplift. The dedicated scale harness
([`internal/bench/fanscale.go`](../../internal/bench/fanscale.go)) prices exactly those
points against the **N=1 single-agent baseline** and is wired to a reproducible command
path. Medians over 16 seeded trials (research-goal, P=2048, sub-turns=4, seed 1,
`fak 0.34.0`, `go1.26.3`); checked-in artifact
[`experiments/fanout/fanscale-d001.json`](../../experiments/fanout/fanscale-d001.json):

| N (sub-agents) | calls | **coord overhead** (turns vs N=1) | coord overhead ÷ baseline | parallel_speedup | **cross_uplift** (measured) | prefix_tokens_saved = (N−1)·P | tax_clawed_back (modeled) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1    | 8    | 0   | 0×    | 1.0  | **0**    | 0          | 0%    |
| 100  | 404  | 4   | 0.67× | 40.6 | **+93**  | 202,752    | 61.6% |
| 500  | 2,004| 24  | 4.0×  | 67.5 | **+495** | 1,021,952  | 61.7% |
| 1000 | 4,004| 49  | 8.2×  | 73.7 | **+990** | 2,045,952  | 61.7% |

Reproduce (deterministic, in-process kernel arithmetic — **no model call**, runs on the
agent host): `go run ./cmd/fanbench --scale --grid canonical --prefix 2048 --trials 16
--seed 1 --out experiments/fanout/fanscale-d001.json`.

Read three things off the acceptance grid:

1. **Coordination overhead vs the N=1 baseline grows with the join, not the work.** The
   single-agent critical path is 6 turns; the fold/synchronous-join tax adds **+4 turns at
   N=100, +24 at N=500, +49 at N=1000** — an **8.2× depth multiple** over baseline by
   N=1000. The lead must wait on the slowest of N sub-agents and then fold N results, so
   added width buys parallel throughput (speedup 1.0→73.7) but pays a critical-path tax that
   keeps rising — the coordination overhead the acceptance criterion names, priced against
   one agent so a "fan-out win" is always budget-controlled.
2. **Cross-agent reuse uplift scales ~linearly with N** (the MEASURED `SHARED − ISOLATED`
   sibling dedup): **+93 / +495 / +990 calls** the interleaved fan-out deletes that the same
   sub-agents run apart cannot. `N=1` is exactly **0** (a lone worker has no sibling to share
   with — the budget-controlled control).
3. **Prefix reuse is exact `(N−1)·P` geometry, not a model** — 202,752 / 1,021,952 /
   2,045,952 prefill tokens the SHARED arm never recomputes, and the modeled prompt-cache
   clawback sits on its **~61.7% plateau** at every acceptance point (P fixed).

**What this grid does NOT do (the honest gate that keeps #255 open).** It is the
**host-computed reuse + coordination geometry**, exactly the measured/modeled split of §1 —
*not* the live-model wall-clock. The **structural** cross-framework comparison (which
orchestrator re-sends what, and the token multiplier that implies) is given in the next
subsection; what stays deferred is the **live-model wall-clock** half. The remaining
D-001 acceptance — **benchmark fak against LangGraph / AutoGen / CrewAI at these N with a
live model** (end-to-end wall-clock) — is a `DeferredRun` (`deferred_run` field in the
artifact): it needs a bench node with the three frameworks installed and a served model,
which the agent host lacks. Running it here would starve the agent seat and skew the
number, so it is named as deferred rather than fabricated (the BENCHMARK-AUTHORITY rule).
That cross-framework task-success axis overlaps **#429** (the §5 controlled litmus) and the
live-model seam **#106**.

### Cross-framework coordination-overhead model — the token multiplier each orchestrator imposes (structural, not wall-clock) (#255)

The D-001 scope names two cross-framework asks — *"Compare against LangGraph / AutoGen /
CrewAI"* and *"Document token multiplier."* The **live-model wall-clock** half of that is
the `DeferredRun` above. This subsection ships the half that **is** host-runnable: a
**structural** comparison of the coordination-token cost each orchestrator imposes **by
construction**, derived from each framework's *documented* multi-agent topology — not a
measured wall-clock, and not a head-to-head throughput claim. It is the same honest
discipline as the committed CrewAI manager-worker model
([`examples/crewai-crew/`](../../examples/crewai-crew/README.md), 4.93×): **deterministic
token arithmetic from a stated geometry**, governed by
[`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md), and it sits inside the
already-declared `[SIMULATED]` reuse claim in `CLAIMS.md` (the win is **reuse-vs-reprefill
over a stack that re-sends the shared prefix per coordination step — the common framework
default — NOT** a win over a tuned shared-prefix engine).

**The one structural fact all three share.** A multi-agent orchestrator advances by
re-invoking a model endpoint, and on every coordination step it re-sends a **stable,
growing shared prefix** — the goal/brief plus the accumulated history. None of the three
implements cross-call **KV-prefix reuse in its own coordination layer**: the prefix is
re-materialized each step unless the *serving backend* deduplicates it (vLLM Automatic
Prefix Caching, SGLang/RadixAttention — both cited at the top of this doc) or the provider
offers prompt caching. `fak` supplies that reuse **kernel-side, independent of backend** —
the measured `(N−1)·P` prefill elision and the ~61.7% clawback plateau in the grid above.

Token accounting at one stated geometry — shared brief **B = 2,000** tokens, per-step
unique instruction **u = 300** tokens, **S = 12** coordination steps (a 3-worker crew over
~4 rounds, the geometry the CrewAI example already uses). "Naive" re-prefills the shared
prefix each step; "reuse" prefills it once and (for the append-only histories) incrementally
caches the growing prefix so each step adds only its new `u` tokens:

| Framework (multi-agent mode) | Documented coordination topology — what is re-sent per step | naive prefill | with prefix reuse | **structural token multiplier** |
|---|---|---:|---:|---:|
| **CrewAI** — hierarchical | manager re-sends the shared brief on every delegation ([docs](https://docs.crewai.com/en/learn/hierarchical-process)) | `S·B + S·u` = 27,600 | `B + S·u` = 5,600 | **4.93×** |
| **AutoGen** — GroupChat | every agent re-reads the full, append-only transcript each turn — brief + growing history ([conversation patterns](https://microsoft.github.io/autogen/0.2/docs/tutorial/conversation-patterns/), [paper](https://arxiv.org/pdf/2308.08155), [growing speaker-selection history](https://github.com/microsoft/autogen/issues/2499)) | `S·B + u·S(S−1)/2` = 43,800 | `B + S·u` = 5,600 | **7.82×** |
| **LangGraph** — StateGraph | each node re-receives the accumulated state; the default `add_messages` reducer is **append-only**, never deleting ([state ref](https://reference.langchain.com/python/langgraph/graph/state), [state management](https://deepwiki.com/langchain-ai/langgraph-101/3.1-state-management)) | `S·B + u·S(S−1)/2` = 43,800 | `B + S·u` = 5,600 | **7.82×** |

Read three things off it:

1. **The token multiplier is the count of coordination steps on the shared prefix.** Without
   reuse, the brief (and, for AutoGen/LangGraph, the append-only transcript) is re-prefilled
   every step — `S×` on the brief, super-linear once the transcript grows. With prefix reuse
   it collapses toward `1×`. This is the "token multiplier" the scope asks to document,
   generalized across the three frameworks.
2. **AutoGen and LangGraph pay more than CrewAI because their history is broadcast/accumulated,
   not just re-briefed.** The full transcript every agent re-reads each turn grows the prefill
   `~S²` (`u·S(S−1)/2`), which is why AutoGen's own docs report a 10-round/4-agent chat landing
   at 40k–80k tokens — the model here lands at 43,800 for S=12, inside that documented band (a
   consistency cross-check, not an asserted measurement).
3. **The frameworks are not the lever; the prefix cache is.** All three reach the "with reuse"
   column **only** through a prefix-caching backend; `fak` is the one that provides it in the
   kernel regardless of provider, so the reuse is the **measured** geometry above rather than a
   backend the operator must hope is enabled.

**Still deferred (unchanged):** the **live-model end-to-end wall-clock** with the three
frameworks installed — token *and* latency, real model — remains the `DeferredRun` named
above (bench-node-gated). This subsection prices the **structural** coordination tax the
frameworks impose by design; it does not assert a live throughput number.

### Prefix-scale impact (new, fixed N=256)

The longer-prefix pscale artifacts keep the fan-out topology fixed (`N=256`,
`sub_turns=4`, `trials=8`, seed `1592652270`) and vary the shared master-goal prefix.
This isolates the model-side question: as the stable prompt grows from smoke-size to
near/full long-context, how much more of the naive N× prefix tax can reuse remove?

_(pscale rows re-run with 8 trials + fixed seed 1592652270; the +247 here is within
±3 calls of the §1 headline's +251 for the same N=256/P=2048 shape — trial-count
variance, not a different result.)_

| shared prefix P | cross_uplift p50 | prefix_tokens_saved | tax_clawed_back | net modeled value/run |
|---:|---:|---:|---:|---:|
| 1,024 | +247 | 261,120 | 47.0% | $0.98 |
| 2,048 | +247 | 522,240 | 61.7% | $1.69 |
| 8,192 | +247 | 2,088,960 | 80.7% | $5.91 |
| 32,768 | +247 | 8,355,840 | 87.4% | $22.81 |
| 131,072 | +247 | 33,423,360 | 89.3% | $90.42 |
| 262,144 | +247 | 66,846,720 | 89.6% | $180.57 |
| 524,288 | +247 | 133,693,440 | 89.7% | $360.86 |
| 1,048,576 | +247 | 267,386,880 | 89.8% | $721.44 |

The impact is clean:

1. **Tool-result dedup is independent of prefix size.** `cross_uplift` stays +247
   because the same stochastic tool-call workload is being replayed; longer prompts do
   not manufacture more real kernel hits.
2. **The prefix lever scales almost linearly in dollars and prefill work avoided.**
   `prefix_tokens_saved = (N−1)·P`, so doubling P doubles the avoided redundant prefix
   materialization.
3. **The tax-clawed-back curve approaches the provider-cache ceiling.** At 1M tokens the
   modeled curve is 89.8%, close to the 90% cached-read discount assumed by the default
   Anthropic-style price model; the remaining gap is cache-write overhead plus suffix,
   decode, and fold tokens.
4. **This is a strategy result, not a real 1M-token wall-clock result.** The pscale row
   prices a byte-identical shared prefix under a transparent cache model; it does not yet
   prove that a selected model/backend can actually hold and prefill a 1M-token context
   within memory or latency targets. That validation is tracked in #104 / #431.

#### Measured-path probe — real wall-clock vs modeled economics (#431)

The rows above are **modeled**. To keep that token economics strictly apart from what a
real model can actually serve, `tools/fanout_longctx_probe.py` interrogates the three
candidate long-context paths on the host and, for each target P, **either measures a real
prefill or records the structured CEILING that stopped it — it never extrapolates the
modeled curve into a wall-clock number.** It sizes the fp16 KV cache from real
`qwen25-7b` geometry (mirrored from `internal/turnbench/longcontext.go`: 28 layers × 4 KV
heads × 128 head-dim) so the memory ceiling is a quantity, not a guess. Artifacts:
`experiments/fanout/pscale/longctx-measure-p262144.json`,
`longctx-measure-p524288.json`, `longctx-measure-p1048576.json`, and the rolled-up
`longctx-measure.csv`.

On the reference host that produced the checked-in artifacts (win32, 253.6 GiB RAM, no
dedicated GPU VRAM) every target P is `SKIPPED_NO_LONGCTX_PATH`:

| candidate path | reason | recorded ceiling |
|---|---|---|
| fak in-kernel | `CONTEXT_CEILING` | selected checkpoint `max_position_embeddings=512` ≪ 262144 |
| llama.cpp | `MODEL_UNAVAILABLE` | `llama-cli`/`llama-bench` present, but no GGUF with context ≥ P on host |
| vLLM / SGLang | `SERVER_UNAVAILABLE` | no prefix-caching server reachable (127.0.0.1:8000 / :30000) |

Two honest facts fall out, neither visible in the modeled table:

1. **The blocker on this host is the missing long-context checkpoint, not memory.** The
   KV cache is 14.0 GiB at P=262144, 28.0 GiB at 524288, and 56.0 GiB at 1048576 — all of
   which **fit** this host's 253.6 GiB RAM (`kv_fits_host_ram: true`). A real 256K–1M
   validation is gated on a checkpoint and a backend that *admit* the context length, not
   on this host's memory.
2. **The skip is recorded, never extrapolated.** `ttft_ms` / `prefill_ms` stay `null` for
   a skipped path; the modeled dollars and tokens-saved live only in the pscale table
   above. Re-run the probe on a host *with* a qualifying path — an in-kernel checkpoint
   whose context ≥ P, a long-context GGUF for llama.cpp, or a running vLLM/SGLang
   prefix-cache server — and the corresponding row flips to `measured` with the wall-clock
   fields filled. The probe embeds only stable host facts (no wall-clock timestamp), so a
   re-run on the same host reproduces the artifact byte-for-byte.

### Alignment with real model / live FAK runs

This fanbench result is aligned with the rest of the shipped evidence, but it sits on a
different axis:

- **Live real-model agent runs (`LIVE-RESULTS.md`) prove the trust floor.** Gemini and
  local Qwen runs show poisoned tool output reaching the unprotected baseline and being
  quarantined by FAK; the weak-model runs show task completion rescued by containment.
  fanbench does not replace that evidence. It says what reuse could save *after* the
  agent trajectory remains valid.
- **Model throughput docs (`MODEL-BASELINE-RESULTS.md`) bound raw compute.** FAK's own
  model path has been made competitive on raw prefill/decode for small/local models, but
  fanbench's long-prefix pscale rows are not a substitute for running a real 256K-1M
  prefill wall-clock on an actual checkpoint/backend.
- **Session value-stack docs (`SESSION-VALUE-STACK-RESULTS.md`) measure reuse-vs-redo on
  live model calls where tractable.** fanbench extends the same reuse story to the
  one-master-goal fan-out topology and very long shared prefixes, but its prefix-scaling
  dollar rows are modeled token economics.
- **Task quality is still separate — and now measured on a litmus axis (#429).** Wider
  fan-out and longer context can save repeated work while still producing duplicated,
  irrelevant, or low-quality subagent output. §5 makes this concrete: a controlled-litmus
  coverage@N / realized@N / matched-budget-control lane
  (`experiments/fanout/taskquality-litmus.json`) shows fan-out saving cost while a
  matched-budget single agent matches or beats it on coverage. The real-model half of the
  seam (#106) stays open.

---

## §2 — The over-claim guardrail (what this is NOT)

The prefix-reuse / `tax_clawed_back` numbers are a **reuse-vs-no-reuse / vs-stateless
ablation**, exactly like `cmd/fleetserve`'s — the win over a consumer that **re-sends the
master-goal prefix per sub-agent** (the common framework default: every sub-agent ships
the full system+goal prompt; or a stateless API consumer that does not use prompt
caching). **It is NOT a head-to-head throughput win over a tuned shared-prefix serving
engine.** SGLang/RadixAttention, vLLM automatic-prefix-caching, and llama.cpp under
`kv_unified`+`seq_cp` **also** prefill a byte-identical prefix once — they occupy the same
lever. fanbench measures *how much of the naive N× prefix redundancy the reuse removes*,
priced at documented Anthropic prompt-cache multiples (read 0.1× / write 1.25×); it does
not claim to beat those engines. The `cross_uplift` is a fak-vs-fak SHARED-vs-ISOLATED
ablation (the fan-out's benefit over running the sub-agents apart), not a vs-competitor
claim.

### Three axes, kept strictly apart — fanbench geometry vs cache pricing vs tuned-engine wall-clock (#430)

The over-claim guardrail above is now backed by a checked-in peer artifact
(`experiments/fanout/peer-baselines.json`) so the three things this result mixes are never
blended into one number. Reference cell `N=256`:

| axis | what it is | value (N=256) | is it a vs-engine claim? |
|---|---|---|---|
| **A — fak-vs-fak reuse geometry** (MEASURED) | the SHARED−ISOLATED path-swap + exact prefill-tokens the kernel never redoes | `cross_uplift = +251` calls; `(N−1)·P = 522,240` prefix tokens saved (P=2048) | **No** — fak's fan-out vs the same sub-agents run apart |
| **B — provider prompt-cache pricing** (MODELED) | the naive-N× tax clawed back at documented cache multiples | `tax_clawed_back = 61.7%` at Anthropic read 0.1× / write 1.25× | **No** — vs a stateless consumer that re-sends the prefix per sub-agent |
| **C — tuned engine wall-clock** (MEASURED, peer artifact) | what an engine that already shares the byte-identical prefix does in wall-clock | llama.cpp `kv_unified`+`seq_cp`: **17.2 vs fak 5.2 agents/s** @ C=32, P=1024, D=32 — *llama.cpp ahead, preliminary*; SGLang RadixAttention: **86.7% agents hit-rate (fak matches), 4.87–5.84× live prefill speedup**, SGLang published up to 6.4× throughput / 3.7× latency (GPU regime); vLLM APC: *not run on this host (SERVER_UNAVAILABLE)* | **Yes** — and it does **not** favour fanbench |

The honest reading of column C: the tuned engines occupy the **same** prefix-reuse lever.
On the one CPU shared-prefix point measured head-to-head, **llama.cpp is ahead of fak**
(17.2 vs 5.2 agents/s, flagged preliminary in
[`LLAMACPP-HEADTOHEAD-RESULTS.md`](LLAMACPP-HEADTOHEAD-RESULTS.md) §Axis 4); against
SGLang/RadixAttention fak **matches the cache-hit-rate** axis
([`RADIXATTENTION-RESULTS.md`](RADIXATTENTION-RESULTS.md)) but does not out-throughput a GPU
serving engine; vLLM's Automatic Prefix Caching was not runnable on this no-GPU host and is
recorded as a ceiling, never extrapolated. So columns A and B (where fanbench's numbers
live) are kept apart from column C (where the engines win or tie) — exactly so no reader
mistakes the reuse-vs-reprefill geometry for a vs-engine throughput win. Full provenance,
the per-C llama.cpp sweep, and the host ceiling that gates a same-host fresh run are in
[`../../experiments/fanout/peer-baselines.json`](../../experiments/fanout/peer-baselines.json).

---

## §3 — The saturation knee (latency, not quality)

`parallel_speedup = total_work_turns / critical_path_turns` rises then **saturates**: 3.0
(N=4) → 29 (N=64) → 57.7 (N=256) → **72.8 (N=1024)**. The N sub-agents run in parallel, so
the wave costs one sub-agent's turns; but the **fold grows with N** (`fold_turns =
⌈N·fold_tokens / fold_budget⌉` — the synchronous-join coordination tax, "the lead waits on
the slowest of N then synthesizes N results"). Past N≈256 the fold dominates the critical
path and added width buys little wall-clock — the unmapped high-N knee, made visible.

This is a **latency/throughput** saturation. The *quality* inversion the literature
reports (realized accuracy peaks then declines with an imperfect verifier; "more agents ≠
better") is a **task-success** phenomenon, on a different axis from this latency knee —
fanbench's cost sweep measures kernel cost, not goal correctness (no ground-truth
sub-results). That quality axis is now its own controlled-litmus lane in **§5** (#429:
coverage@N / realized@N / matched-budget control with ground-truth atoms), which shows the
realized-accuracy inversion the latency knee here cannot. See also the research brief §5–6
for the published anchors and the remaining seams (MAST failure tagging, hierarchical
depth>1, parallel-vs-sequential at matched budget).

---

## §4 — Profiles & determinism

- **`research-goal`** (headline): read-heavy, no writes — cross-agent dedup cleanly
  positive and saturating.
- **`write-goal`**: `p_write=0.25` — sub-agent writes bump the global world and clear every
  sibling's warmed reads, so `cross_uplift` is throttled and can go **negative** (the
  honest invalidation tension, identical to `fleetbench`'s write-heavy finding).
- **`no-share`** (anti-inflation control): no shared reads, no plan — `cross_uplift` is
  **identically 0 at every N**; a non-zero value would be a harness bug (asserted by
  `TestFanoutNoShareZeroUplift`).

A fixed `(profile, N, sub-turns, trials, seed)` yields the identical surface
(`TestFanoutDeterministic`). The model-backed witness (`TestPrefixReuseFanoutWitness`)
asserts all N clones are **bit-identical** to an independent full prefill — cross-agent
prefix reuse provably never changes a sub-agent's result.

Prefix options are now first-class in the artifact: CSV rows include `prefix_tokens`,
JSON cells include `prefix_tokens`, and JSON sweeps include `prefix_grid`. `-prefix`
keeps the old single-P run. `-prefixes smoke,small,medium,long,big` expands to
1K/2K/8K/32K plus a near-full-context prefix; `-prefixes all` is the same range. The
`big`/`max` preset is computed as `model_context - suffix - max(sub_turns)*decode`, using
`-model-config` (`max_position_embeddings` / `model_max_length`) when provided, otherwise
`-model-context` (default 131,072).

## §5 — Task-quality litmus: does fan-out improve the ANSWER? (#429)

Everything above prices **kernel cost geometry** — reuse, dedup, prefix economics,
latency. It is silent on the question a reader actually cares about: does fanning wider
make the *answer better*? The research brief (§8) lists "real task success / coverage@N /
realized@N" as an explicit out-of-scope seam precisely because cost geometry and task
quality are **different axes** — and blending them is the most common multi-agent
benchmark error. This section keeps them apart on purpose.

`tools/fanout_taskquality.py` runs a **controlled litmus** task suite over the same flat
fan-out: one master goal whose complete solution is a known ground-truth set of `G=12`
sub-result *atoms*. N sub-agents (4 sub-turns each, homogeneous pool) draw atoms, an
imperfect verifier/fold selects from their outputs, and a fraction are fed an adversarial
tool result. Each task row is **joined to the real fanbench cost cell for the same N**
(`coverage`/`realized` next to the MEASURED `cross_uplift` / `tax_clawed_back` /
`parallel_speedup`), so the artifact literally connects a litmus task run to the
fanbench-like N grid. Artifact: `experiments/fanout/taskquality-litmus.json` (+ `.csv`),
64 seeded trials, medians. Reproduce / gate (byte-identical re-run, same discipline as the
pscale probe): `python tools/fanout_taskquality.py --check`.

> **HONEST FRAME — this is `[SIMULATED]`, NOT a real-model run.** The task *outcomes* are
> a transparent knobbed model grounded in published anchors — homogeneous pools saturate
> ~4 agents (Agent Forest, [arXiv:2402.05120](https://arxiv.org/html/2402.05120v1));
> imperfect-verifier realized accuracy peaks then declines, compute-optimal K≤5
> ([arXiv:2411.17501](https://arxiv.org/html/2411.17501v1)); step-repetition is the single
> most frequent MAST failure mode at 15.7% ([arXiv:2503.13657](https://arxiv.org/abs/2503.13657));
> naive MAS sits ~33–59% correct. The injection arms are grounded in the **real** quarantine
> evidence in [`LIVE-RESULTS.md`](LIVE-RESULTS.md) (Gemini / local Qwen: poisoned tool output
> reaches the naive baseline; fak quarantines it). The cost columns are joined **verbatim**
> from the measured `fanbench-research.csv`. This lane proves the cost/quality **separation**;
> it does **not** prove real-model quality.

Medians over 64 trials (`experiments/fanout/taskquality-litmus.csv`):

| N | coverage@N | realized@N | verifier_success | duplicate_work | inj_contained fak/naive | **matched-budget single-agent coverage** | (joined) tax_clawed_back | (joined) parallel_speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1   | 0.083 | 0.083 | 1.00  | 0.00 | 1.00 / —    | **0.250** | 0%    | 1.0   |
| 4   | 0.250 | 0.167 | 1.00  | 0.25 | 1.00 / 0.00 | **0.500** | 54.8% | 3.0   |
| 16  | 0.583 | 0.500 | 1.00  | 0.53 | 1.00 / 0.00 | **0.833** | 60.4% | 11.0  |
| 64  | 0.917 | **0.667** | 1.00  | 0.82 | 1.00 / 0.00 | **1.000** | 61.4% | 29.0  |
| 256 | 1.000 | 0.583 | 0.875 | 0.95 | 1.00 / 0.00 | **1.000** | 61.7% | 57.7  |

Read five things — and note that **none of them is "wider fan-out makes a better answer"**:

1. **Coverage@N grows, then SATURATES.** Distinct ground-truth atoms reached climbs
   log-linearly (0.08 → 1.0 by N≈64) and then can rise no further — the homogeneous-pool
   knee the literature reports at ~4 agents, made visible against the cost curve.
2. **Realized@N PEAKS then DECLINES.** The verifier/fold is imperfect: by N=64 it surfaces
   66.7% of atoms, but by N=256 the decoy pile from failed sub-agents crowds correct atoms
   out of the finite fold, dropping realized to 58.3% and verifier precision to 0.875. More
   agents past the knee make the *delivered* answer **worse**, not better — exactly the
   inversion §3's latency knee could not show.
3. **Duplicate work explodes.** 25% of competent outputs re-derive a covered atom at N=4,
   95% at N=256 — the step-repetition failure mode. Wider fan-out buys mostly redundant work.
4. **Injection containment is the real, decisive split.** Under the fak arm every
   injection-exposed sub-agent is quarantined (1.00); under the naive arm none is (0.00 at
   N≥4), and the lost work drags naive-arm coverage below fak's — the litmus echo of the
   real weak-model rescue in `LIVE-RESULTS.md`. (At N=1–2 the naive cell is `—`/vacuous:
   too few trials draw an injection to be meaningful.)
5. **The matched-budget single-agent control matches or BEATS the fan-out at every N.**
   Given the fan-out's *total* call budget as one sequential trajectory (no cross-agent
   redundancy, no parallelism), the lone agent's coverage is **higher** at every width —
   0.50 vs 0.25 at N=4, 0.83 vs 0.58 at N=16. This is the research brief's budget-controlled
   design law: **at equal compute, fanning out does not win on task quality.**

> **The one-line honest reading.** Fan-out's wins in §1–§3 are real and they are
> **cost/latency/reuse** wins — the joined columns show 62% of the token tax clawed back
> and a 57× parallel speedup at N=256. They are **not** quality wins: at matched budget a
> single agent covers as much or more, realized accuracy inverts past the verifier knee,
> and most of the added width is duplicate work. The kernel makes wide fan-out *cheaper to
> run*; it does not make the *answer better*. The only quality axis where fak changes the
> outcome here is **injection containment**, and that is a safety-floor result, not a
> fan-out-width result.

**What is still open (the real-model half).** This is a controlled litmus, not a live
model sweep. A real `fak agent` fan-out across the N grid with real ground-truth tasks
(the seam #429 also names) needs a model/host budget this no-GPU node does not have; the
single live A/B that exists today (`LIVE-RESULTS.md`) is single-task, single-agent. The
litmus pins the **metric definitions and the cost/quality separation**; flipping it to a
measured real-model row is the open work.

## §6 — Tracked limitations and next steps

The limitations are now explicit GitHub issues rather than hidden caveats:

- #104 / #431 — validate the 256K-1M prefix-scale points with real long-context model
  wall-clock, KV memory, and failure ceilings. **Fail/skip half landed (#431):**
  `tools/fanout_longctx_probe.py` records a structured per-path ceiling instead of
  extrapolating the modeled curve (artifacts under `experiments/fanout/pscale/`, see the
  measured-path probe subsection in §1). **Open:** a host with a checkpoint/backend that
  admits a 256K+ context, to flip a row from `skipped` to `measured` wall-clock.
- #430 — add tuned shared-prefix serving baselines for vLLM/SGLang/llama.cpp so this
  remains an honest reuse-vs-reprefill claim, not an unproven vs-engine claim.
  **Addressed:** the three-axis table in §2 plus the checked-in peer artifact
  `experiments/fanout/peer-baselines.json` pin column C (tuned-engine wall-clock) to the
  shipped RadixAttention head-to-head and the measured llama.cpp `kv_unified`+`seq_cp`
  C-sweep, with vLLM recorded as a host ceiling. **Open:** a same-host fresh run of all
  three on one box (no small GGUF / no GPU / no prefix-cache server here).
- #429 / #106 — align fanbench cost curves with real task success and the live `fak agent`
  evidence: coverage@N, realized@N, verifier success, duplicate-work rate, injection
  containment, and matched-budget single-agent controls. **Litmus half landed (#429):**
  `tools/fanout_taskquality.py` runs a controlled ground-truth task suite over the fan-out
  N grid and joins the task metrics to the measured cost cells, with a matched-budget
  single-agent control proving fan-out saves cost but not quality (§5; artifact
  `experiments/fanout/taskquality-litmus.json`, gated by `--check`). **Open (#106):** a
  real-model `fak agent` fan-out across the N grid — needs a model/host budget this no-GPU
  node lacks.
- #107 — make fanout plot regeneration reproducible; the 262K/524K/1M CSV/JSON
  artifacts exist, but this local node lacked `matplotlib`, so the PNGs were not
  regenerated in this pass.
