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
   within memory or latency targets. That validation is tracked in #104.

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
- **Task quality is still separate.** Wider fan-out and longer context can save repeated
  work while still producing duplicated, irrelevant, or low-quality subagent output. A
  real task-success / coverage@N lane is tracked in #106.

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
better") is a **task-success** phenomenon and is **out of scope** here — fanbench measures
kernel cost, not goal correctness (no ground-truth sub-results). See the research brief §5–6
for that axis and the documented future seams (coverage@N / realized@N, MAST failure
tagging, hierarchical depth>1, parallel-vs-sequential at matched budget).

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

## §5 — Tracked limitations and next steps

The limitations are now explicit GitHub issues rather than hidden caveats:

- #104 — validate the 256K-1M prefix-scale points with real long-context model
  wall-clock, KV memory, and failure ceilings.
- #105 — add tuned shared-prefix serving baselines for vLLM/SGLang/llama.cpp so this
  remains an honest reuse-vs-reprefill claim, not an unproven vs-engine claim.
- #106 — align fanbench cost curves with real task success and the live `fak agent`
  evidence: coverage@N, realized@N, verifier success, duplicate-work rate, and
  matched-budget single-agent controls.
- #107 — make fanout plot regeneration reproducible; the 262K/524K/1M CSV/JSON
  artifacts exist, but this local node lacked `matplotlib`, so the PNGs were not
  regenerated in this pass.
