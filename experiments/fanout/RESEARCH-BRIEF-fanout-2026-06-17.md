# Research Brief — Designing a 1-Goal → N-Subagent Fan-Out Benchmark for FAK

> **Provenance.** Produced 2026-06-17 by a 19-agent research workflow (6 parallel
> survey angles → 12 adversarial verification passes → 1 synthesis). Verdict tally:
> **11 claims confirmed, 0 refuted, 1 uncertain** (CrewAI overhead figures — flagged
> inline). Figures tagged `[confirmed]` were checked verbatim against a primary
> source. This brief is the design input for `cmd/fanbench` + `internal/turnbench/fanout.go`.

**Scope.** Grounds a new FAK benchmark component that models **one master goal that
spawns hundreds-to-thousands of sub-agents**, whose core lever is **cross-agent
shared-prefix KV reuse + cross-agent tool-result dedup**.

---

## 1. The fan-out regime — topologies and precise vocabulary

A "fan-out" is an orchestrator decomposing one goal, spawning N workers, and folding their outputs.

- **Orchestrator-worker (lead/subagent).** One central agent decomposes the goal and directs N workers that each execute a slice, then synthesizes results. Anthropic's canonical instance: a **LeadResearcher** (Opus 4) plans, **spins up 3–5 subagents (Sonnet 4) in parallel**, and runs a separate **CitationAgent** fold; execution is **synchronous** (lead waits for each wave). `[confirmed]` (https://www.anthropic.com/engineering/multi-agent-research-system)
- **Supervisor.** A single orchestrator decides *who-goes-next* over shared state until termination (LangGraph). `[confirmed]` (https://github.com/langchain-ai/langgraph-supervisor-py)
- **Hierarchical (supervisor-of-supervisors).** Multi-level. The recommended way to scale **beyond ~5 workers is to add depth, not widen one parent** — to avoid context fragmentation. `[confirmed]`
- **Network / swarm.** Fully-connected mesh; O(N²) communication paths make it viable only at small fixed counts (2–4). `[confirmed]`
- **Map-reduce / pipeline.** Fixed-role assembly line (MetaGPT). Fan-out by *role diversity*, not *replica count*. `[medium]` (https://arxiv.org/abs/2308.00352)
- **Debate (MAD).** N agents critique over R rounds; at matched compute rarely beats CoT+self-consistency. `[high]` (https://arxiv.org/abs/2305.14325)
- **Ensemble / best-of-N / self-consistency.** N independent samples + a **fold** (majority vote / reward-model / judge). The simplest, strongest, cheapest parallel baseline. `[high]` (https://arxiv.org/pdf/2203.11171, https://arxiv.org/abs/2407.21787)
- **Mixture-of-Agents (MoA).** Layered proposers → aggregator; **Self-MoA** (one strong model) often wins by ~3.8%. `[high]` (https://arxiv.org/abs/2406.04692)

**Two terms to keep separate:**
- **Fan-out / fan-in (fold).** Fan-out = spawn N; fold = synthesize. The fold reconciles conflicting/redundant sub-results and is the dominant failure site.
- **Critical path (depth) vs total work (sum).** Wall-clock for parallel fan-out = `max over branches + merge`, **not** the sum. Cost/throughput is governed by the *sum*. Conflating them is the most common benchmark error. `[high]`

---

## 2. How many sub-agents in practice — and what bounds the count

| Source | Fan-out width | Notes |
|---|---|---|
| Anthropic Research (complexity-tiered) | **1 / 2–4 / >10** | simple=1 agent; comparison=2–4; complex=">10 with divided responsibilities" |
| Anthropic sweet spot | **3–5 parallel** | each using 3+ tools in parallel |
| Claude Code (community-reported) | **~10 concurrent**, then queue | not official |
| Magentic-One | **exactly 4** | fixed roles; stall counter ~2 → replan `[confirmed]` (https://arxiv.org/html/2411.04468v1) |
| **Spawning 50 for a simple query** | **failure mode** | a pathology the lead's prompt had to suppress |

**What bounds N today (the exact pressures FAK exists to measure):** ~15× token cost of multi-agent vs chat (4× single) `[confirmed]`; context fragmentation; weak real-time coordination (36.94% coordination failures across AutoGen/CrewAI/LangGraph); synchronous join (lead waits on slowest of N); coordination overhead grows O(N²). Scaling move beyond ~5 is **hierarchical, not wider**. **Bottom line: production runs N=3–10; 10–20 via hierarchy; N≥50 is the pathological, unmeasured regime — exactly the space `fanbench` should map.**

---

## 3. What existing benchmarks measure — and THE GAP

Single-agent task suites (no fan-out axis): **GAIA** (0/1 exact-match, ~15% vs ~92% human), **AgentBench** (8 envs), **τ-bench/τ²-bench** (state-delta grading; introduced **pass^k**; GPT-4o retail pass^1=61.2% → pass^8≈25%), **SWE-bench Verified** (resolved rate), **WebArena/OSWorld/AppWorld/GTA/AssistantBench/MLE-bench**, **TheAgentCompany** (multi-agent = scripted NPCs, not a worker fleet).

Explicitly multi-agent — but tiny: **MultiAgentBench** (ablates 1/3/5/7 agents), **AgentsNet** (~100 *abstract* agents on synthetic graphs — frontier models **collapse to ~0 at 100 agents**; states verbatim *"existing multi-agent benchmarks cover at most 2-5 agents"*). `[high]` (https://arxiv.org/html/2507.08616v1, https://arxiv.org/abs/2503.01935)

> **THE GAP.** No standard benchmark stresses hundreds-to-thousands of parallel
> sub-agents on ONE concrete goal. Mainstream suites are single-agent; explicit
> multi-agent benchmarks top out at 5–7 collaborating agents. Nobody publishes
> throughput/cost/quality curves at N=20/50/100/1000. Serving benchmarks
> (vLLM/SGLang) measure raw request concurrency but are **not tied to a single
> collaborative goal**. **Shared-prefix KV reuse across N siblings — the single
> biggest kernel-level saving — is never quantified for this topology.**

---

## 4. Metrics that matter

| Metric | Definition | Published anchors |
|---|---|---|
| **Success / pass^k** | pass@k (≥1 of k) AND pass^k (ALL k succeed — reliability) | τ-retail pass^1=61.2% → pass^8≈25% |
| **Token multiplier** | total fan-out tokens ÷ single-chat; decompose into shared-prefix vs suffix, cache_read vs creation vs uncached | 4× single, **15×** multi-agent `[confirmed]` |
| **Critical-path latency (depth)** | `max over branches + merge`; user-felt time | parallel 21.3s vs sequential 38.7s (1.82×) |
| **Total work (sum)** | Σ branch + orchestrator; governs $ and throughput; report **separately** | — |
| **Throughput (agent-turns/sec)** | completed agent turns / wall-sec | KVFlow up to 2.19× concurrent |
| **Prefix-reuse ratio** | `shared_prefix_tokens / mean_subagent_input_tokens`; (1−ratio)=suffix | **0.8 to >0.99** when goal dominates `[medium]` |
| **Effective cached-input cost** | `1.25×(write) + N·0.1×(reads)` vs naive `N·1.0×`; → ~10× prefix savings at large N | read 0.1× / write 1.25–2× (Anthropic) `[confirmed]` |
| **Cost-of-pass** | total cost / tasks actually solved (NOT cost-per-attempt) | SWE-Agent+GPT-4: $0.24/instance but **$32.5 per fix** |
| **Coordination overhead / error-amplification** | orchestrator tokens / total; error multiplier by topology | turns `T=2.72·(n+0.5)^1.724` (super-linear) `[high]` (https://arxiv.org/abs/2512.08296) |
| **Wasted-token fraction** | redundant/reclaimable tokens | **~29.68%** on GAIA pass@1 `[medium]` |
| **Cost-quality Pareto** | accuracy vs FLOPs/$ (NOT accuracy vs N) | smaller+more-samples often dominates |
| **Saturation/knee** | the N where added agents stop helping | homogeneous pools saturate **~4 agents** |

> CrewAI "+30–50% manager overhead" `[UNCERTAIN — single non-authoritative blog]`. Do not anchor FAK targets on these.

---

## 5. Scaling behavior and laws

- **Coverage@N (oracle pass@k)** grows **log-linearly (exponentiated power law)** over ~4 orders of magnitude: SWE-bench Lite **15.9%@N=1 → 56%@N=250**; GSM8K/MATH **>95%@N=10,000**. `[high]` (https://arxiv.org/abs/2407.21787)
- **Selection bottleneck.** Coverage converts to realized accuracy only with a verifier; majority vote plateaus after a few hundred samples. Judge WR 0.810 vs majority 0.496 vs synthesis **0.179 (worse than nothing)**. `[medium]`
- **Sampling-and-voting (Agent Forest):** gains **saturate past ~4** for homogeneous (correlated) pools. `[high]` (https://arxiv.org/html/2402.05120v1)
- **Imperfect verifier inverts the curve:** realized accuracy peaks then declines; compute-optimal **K≤5**. `[high]` (https://arxiv.org/html/2411.17501v1)
- **Best-of-N Goodhart hump:** true reward peaks then falls; effective KL ~log(N). `[high]`
- **Compute-optimal inference:** smaller-model+more-samples often beats larger+fewer at fixed FLOPs. `[high]` (https://arxiv.org/abs/2408.03314)
- **Parallel often Pareto-dominated by sequential** at matched compute (sequential beat parallel in 95.6% of configs). `[medium]`

---

## 6. Failure modes — "more agents ≠ better"

**MAST taxonomy** (Cemri et al., NeurIPS 2025; 1,600+ traces, κ=0.88) `[high]` (https://arxiv.org/abs/2503.13657):
- **FC1 Specification/Design — 43.7%**: step repetition (**15.7%, most frequent mode**), spec disobedience (11.8%), termination-unaware (12.4%).
- **FC2 Inter-Agent Misalignment — 32.15%**: reasoning-action mismatch (13.2%), task derailment (7.4%), ignored input.
- **FC3 Task Verification — 24.5%**: premature/incomplete/incorrect verification.

- **SOTA MAS fail 41%–86.7%** of tasks; treat **~33–59% correctness** as the realistic naive-fan-out baseline. `[high]`
- **At equal token budget, a single agent matches/beats sequential multi-agent** (DPI: agent-to-agent comms is a lossy channel). `[high]` (https://arxiv.org/html/2604.02460v1)
- **Coordination crossover ~45% baseline accuracy**: above it, coordination gives negative returns. `[high]`
- Anthropic's own failures: subagents duplicating work, 50-subagent over-spawn.

> **Design law:** any FAK "fan-out win" must be **budget-controlled** — hold per-agent and total compute constant against the single-agent control, or the win is just extra compute.

---

## 7. Serving infra — why 1-goal→N-subagents is the IDEAL KV-reuse case

N near-identical prompts that diverge only in a short per-agent suffix.

- **Theoretical:** prefix prefill compute + KV memory go **O(N) → O(1)**; savings → **(N−1)/N**. Headline: **prefix-redundancy-eliminated = (N−1)·prefix_tokens.** `[medium]`
- **RadixAttention / SGLang:** automatic prefix reuse; **up to 5×/6.4×** throughput; hit rates 50–99%. `[high]` (https://www.lmsys.org/blog/2024-01-17-sglang/, https://arxiv.org/pdf/2312.07104)
- **Hydragen (the ceiling):** batches prefix-attention as a GEMM; **up to 32× throughput**, growing with batch size AND prefix length; lengthening shared prefix 1K→16K costs Hydragen **<15%** vs naive **>90%**. `[high]` (https://arxiv.org/abs/2402.05099)
- **vLLM PagedAttention + APC:** 14–24× vs naive HF; APC gain **monotonic in shared-prefix ratio (13%→32% at ratio 0.9)**. `[high]` (https://arxiv.org/pdf/2309.06180)
- **KVFlow** (NeurIPS 2025): workflow-aware cache, **up to 2.19× concurrent**. `[high]` (https://arxiv.org/abs/2507.07400)
- **SwarmKV:** snapshot sharing, **~1.95× end-to-end, ~52× lower per-branch activation latency**. `[high]`
- **Provider prompt-cache prices the lever:** Anthropic read 0.1× / write 1.25–2×; OpenAI 0.5×. **Hard constraint: the master-goal prefix must be byte-identical across all N subagents.** `[confirmed]`

---

## 8. Design recommendations for `fanbench` (as implemented)

**Name:** `fanbench` — the fan-out kernel benchmark (`cmd/fanbench`, harness `internal/turnbench/fanout.go`).

**Topology (v1):** master-goal prefix + flat fan-out of N sub-agents + a fold; `depth` reserved for hierarchical. Sweep **N = 1,2,4,…,1024**.

**Mapping onto FAK's REAL kernel levers (verified present):**
- **Shared-prefix KV clone → `model.NewBatchFromPrefix(prefix, n)`** (`internal/model/batch.go:239`): prefix prefilled once + n cheap `KVCache.Clone`s; bit-identical (R14 property, `internal/model/kvreuse_test.go:TestKVPrefixReuseMatchesRecompute` — "clone the cache, prefill only the SUFFIX, skipping the prefix's prefill FLOPs"). **This IS the SHARED arm**; ISOLATED = n independent prefills.
- **Cross-agent tool-result dedup → vDSO tier-2** (`internal/vdso/neardup.go:SetNearDup`): exact key default; near-dup collapses formatting variants; negatives never near-dup-shared.

**What makes it honest (and what `fanbench` does):**
1. **Real kernel events for dedup** — the SHARED-vs-ISOLATED uplift is measured `k.Syscall` tier-2 hits (the `fleetbench` discipline), not arithmetic.
2. **Exact geometry for prefix reuse** — `prefix_tokens_saved = (N−1)·prefix_tokens`, grounded in the shipped `NewBatchFromPrefix` prefill-once property (wall-clock-witnessed by `cmd/fleetserve`); a model-backed witness test asserts N clones are bit-identical so reuse never changes results.
3. **Transparent, knobbed cost model** for the modeled token-multiplier / critical-path-vs-total-work / throughput projections — clearly labeled MODELED, separate from the measured halves.
4. **Determinism** — fixed (profile, N, subturns, trials, seed) → identical surface; `Dist` order-stats over seeded trials.
5. **Budget-controlled** — every sweep includes the N=1 single-agent control; the token-multiplier and tax-clawed-back are reported relative to it; a no-share control gives exactly 0 uplift.
6. **Saturation, not point estimates** — the latency knee (synchronous-join fold cost grows with N while per-branch work amortizes) is reported across the N curve.

**Out of scope for v1 (documented seams, future axes):** real task success / coverage@N / realized@N (needs synthetic goals with ground-truth sub-results — quality inversion is a *task* phenomenon, not a *kernel-cost* one), selector/verifier curves, MAST FC1/FC2/FC3 tagging, hierarchical depth>1, parallel-vs-sequential at matched budget.

**One-line thesis:** *`fanbench` is the first benchmark to sweep N from 1 to 1000+ on a single goal and report, from real FAK kernel events, exactly how much of the 15× multi-agent token tax cross-agent shared-prefix KV reuse + tool-result dedup claws back — and where, in the unmapped high-N regime, the synchronous-join fold cost overtakes the parallel win.*
