---
title: "Borrow study: benchmark integrity + serving metrics — SemiAnalysis InferenceMAX/InferenceX vs fak's witnessed frontier (2026-07-13)"
date: 2026-07-13
kind: borrow-study
source: "SemiAnalysis InferenceMAX / InferenceX — public benchmark repo, cloned to scratchpad for this study (2026-07-13). Exact upstream SHA not pinned here: the scratchpad git volume stalled during write-up; pin it in any issue this note spawns."
license: "Study read only. No InferenceX code was copied; every candidate below transfers as design/methodology, not code (their harness is Python + GitHub Actions; fak is Go+Rust)."
---

# Borrow study: InferenceMAX/InferenceX — perf-benchmark integrity, serving metrics, SLO/goodput, run-reuse

Deep-read of **SemiAnalysis's InferenceMAX/InferenceX** (a continuous, Day-0 LLM-serving
benchmark for token-factory buyers) against fak's already-shipped frontier on the axes it
overlaps: serving telemetry, SLO/goodput, decode-latency methodology, benchmark-run reuse,
perf-benchmark anti-gaming, and the fused fast-but-wrong gate.

- **Method:** 5-way reader fan-out over the study repo (perf-changelog gate; SLO experiment +
  worldview; reuse gate; agentic workload; bench-client + metrics/power), then per-candidate
  **witness on-axis against fak's tree** → route to an existing issue, file only the net-new,
  or document as a validated divergence / already-shipped convergence.
- **fak-side witnessing done this session:** `internal/gateway/serving_metrics.go`,
  `internal/antipattern/checker_games.go`, `internal/compute/decode_throughput.go`, plus an
  **8-agent adversarial establish→refute verification** (2026-07-13) of the four net-new candidates
  against the live tree — every "gap" was independently re-derived and then attacked by a skeptic
  before being trusted (results below reflect the post-refute verdicts, which corrected two axes).
- **Dedup caveat (important):** the first-pass backlog dedup used `gh issue list --search`, which
  **silently returns global cross-repo results** — the serving issues it surfaced (`#128` serving
  surface, `#104` parity harness, `#466` SLO report, `#183` "controlled-reuse") are **leaks from a
  sibling repo, not `anthony-chaudhary/fak`**. Re-run authoritatively via the search API with a
  `repo:` qualifier, the real fak homes are: **#43** (TTFT/TPOT/ITL/goodput serving surface, *closed*),
  **#44** (vLLM/SGLang/native parity harness, *closed*), **#2242** (SLO/goodput P/D planner, *open*),
  **#3079** (GLM-5.2 continuous-batching concurrency sweep + p50/p95 TTFT, *open*), **#4582** (quality
  SLOs/error budgets for serving, *open*). fak's own `#128`/`#104`/`#183` are unrelated docs issues.

## Headline

Like the plano study (`BORROW-ROUTING-SIGNALS-GATEWAY-PLANO-STUDY-2026-07-13.md`), this is a
**convergent-frontier** study. The overlapping surface is **well covered by the existing backlog**
(#43 shipped the TTFT/TPOT/ITL/goodput serving schema, *closed*; #44 the vLLM/SGLang/native parity
harness, *closed*; #2242 a live SLO/goodput P/D planner; #3079 a live concurrency-sweep + TTFT
benchmark; #4582 quality SLOs for serving; #4367 the 24h four-resource GPU campaign; #2236 the
superset epic). On the doctrinal axes fak has **already shipped the mechanism** (witness-not-self-report;
fused fast-but-wrong gate) or made a **deliberate opposite choice** (measure prefix-cache reuse, don't
erase it). The study turned up **three net-new refinements** (verdicts below are post-adversarial-refute):

1. **SLO-conditioned goodput** (joint ttft∧tpot∧e2e) — **REAL but a refinement, not greenfield.**
   fak already computes a **single-e2e-SLO** `GoodputRPS` (webbench `serving.go:685`, `--slo-ms`) *and*
   a success-rate `Goodput` (`serving_metrics.go`); what is genuinely missing is the **joint** gate
   (good iff ttft ∧ tpot ∧ e2e all met). → routes to the live **#2242** (SLO/goodput planner).
2. **Token-position-swept decode SLO + SLA frontier** — **REAL** (fak's *own* docs already admit it:
   `industry-scorecard/serving.md` "has not recorded a latency-vs-throughput Pareto frontier";
   `cost.md` "ships no goodput-under-SLA number"). No context-position TPOT sweep, no SLO-gated
   per-GPU frontier. → routes to the live **#3079** (concurrency sweep) as the natural fold-point.
3. **Benchmark-run reuse by lineage** — **REAL and fully uncovered** (repo-scoped search for
   `benchruns lineage` / `reuse benchmark run` returned zero fak issues). fak stamps the lineage key
   and can even detect *staleness* but never consumes it to *skip* a redundant re-run.
   → **FILED this session as [#4600](https://github.com/anthony-chaudhary/fak/issues/4600).**

---

## The witnessed matrix

### Axis 1 — Serving goodput semantics (REAL refinement — the *joint* gate → route #2242)

**InferenceX mechanism.** Goodput is **SLO-conditioned**: a request counts as "good" only if it
meets **all** of {ttft, tpot, e2el} SLOs simultaneously (`benchmark_serving_random.py:263-266`,
DistServe-style). Fast-but-tail-violating is *not* good. TPOT is computed **decode-only** —
`(latency − ttft)/(output_len − 1)` (`benchmark_serving_random.py:239-240`) — so per-token latency
excludes prefill.

**fak witness (verified — establish PARTIAL, refuter could not overturn).** fak already has **two**
goodput notions, and the adversarial re-check found a *third, stronger* one the first pass missed:
- **success-rate** `Goodput` — `serving_metrics.go` (`:76`, family `:152`): native `turns/decodeSecs`
  (`:189`), scrape `successTotal delta/interval` (`:526`); no latency thresholding.
- **single-e2e-SLO** `GoodputRPS` — webbench `FoldServingSamples` (`serving.go:673-719`) **does**
  SLO-condition, but on **one** threshold: `good++ iff EndToEndMillis ≤ slo` (`:685`); per-request
  TTFT (`:688`) and TPOT (`:692`) are collected as quantiles but **never gate the good count**. One
  `--slo-ms` knob (`cmd/fak/webbench.go:350`).
- the autoscaler `PoolObjective` carries `GoodputTarget` + `TTFTSLO`/`TPOTSLO` (`serving_autoscaler.go:165,170`)
  but consumes them as **independent OR-ed** scaling-breach reasons, never a per-request joint test.

So the earlier "fak's only goodput is success-rate" framing **understated** the machinery. The
genuinely-missing piece is narrow: a request counted good **iff it meets ttft ∧ tpot ∧ e2e
simultaneously** (the DistServe/vLLM definition).

**Verdict → route to the live #2242 (SLO/goodput P/D planner).** The net-new is a small refinement of
the *existing* webbench `GoodputRPS`: replace the single `EndToEndMillis ≤ slo` test with a joint
ttft ∧ tpot ∧ e2e gate (add `--ttft-slo-ms`/`--tpot-slo-ms` alongside `--slo-ms`). Ship-alone; it
sharpens the "aligns with vLLM" claim that today is only nominal. (#43 shipped the underlying schema
but is closed; #2242 is the live consumer of SLO/goodput.)

### Axis 2 — Position-dependent decode SLO + SLA frontier (ABSENT, verified → route #3079)

**InferenceX mechanism.** Decode latency is treated as a **function of token position**, not a
single number: hold OSL small (128), sweep ISL 1024→16384, and plot TPOT against a relabelled x-axis
literally named "Token Position" (`bmk_kimi_k2_sbatch.sh:18`, `plot_sla_frontier.py:95`). For each
position they compute an **SLA frontier**: `max(throughput_per_gpu for runs whose latency < SLO)`
across a **geometric concurrency sweep** (conc ×2, `generate_sweep_configs.py:432-438`) — the max
GPU-normalized throughput still meeting each TTFT/TPOT threshold
(`plot_sla_frontier.py:42-56`, threshold ladders `:17-18`). Steady-state fidelity via
warmup-then-measure at target concurrency (`benchmark_serving_random.py:322-343`).

**fak witness (verified — establish ABSENT, refuter tried both halves and failed).** Both halves are
genuinely absent; the building blocks exist but are never assembled:
- *(a) position-swept decode TPOT:* `decode_throughput.go` is explicitly **P-independent**
  (single-aggregate roofline, `:28-29`); `glmdsatput`/`modelbench`/`q8bench` fold to **one** median
  ms/tok at a fixed prompt; `glm52prefillsweep` sweeps prompt length but measures **PREFILL** (TTFT),
  not decode; `quality/context_cliff.go` is per-position but measures **quality**, not latency.
- *(b) SLA frontier:* `loadgen` sweeps concurrency (`:221`) but is **non-streaming** (`Stream:false`,
  `:114`) so has no TTFT/TPOT, no SLO gate, no per-GPU norm; webbench folds a **single** e2e-SLO at a
  **single** concurrency (`:685,:715`). No "max tok/s/GPU among runs meeting a TTFT/TPOT SLO" fold.
- **fak's own docs already name both as gaps** (`industry-scorecard/serving.md`, `cost.md`).

**Verdict → route to the live #3079** (GLM-5.2 continuous-batching concurrency sweep 1→128 + p50/p95
TTFT). That issue already builds the concurrency-sweep substrate; the net-new *methodology* is to fold
its points through an SLO gate + per-GPU normalization (and capture streaming TPOT-by-position). This
is the difference between an honest and a headline decode number.

### Axis 3 — Benchmark-run reuse by lineage (ABSENT, verified → FILED #4600)

**InferenceX mechanism.** Re-running an unchanged benchmark on scarce GPUs is avoided by a
reuse gate: a passing sweep/eval is reused iff it is keyed to a **commit-SHA identity** that is
still in the PR, gated on artifact-existence and an explicit human authorization
(`/reuse-sweep-run`). Reuse is the throughput lever on a shared cluster.

**fak witness (verified line-by-line — establish ABSENT, refuter confirmed).** fak stamps the lineage
key (`benchcli.Stamp`, `benchcli.go:208-217`; runID from utc+commit+harness `:739`) and can even
detect *staleness* — but the two are never joined into a skip:
- `benchloop.chooseAction` (`benchloop.go:340`), the launch gate, branches only on
  `Catalog.Error`/`RunCount==0`/`Local.Error`/`Local.HasNext`; catalog used for **status counts only**
  (`:247-266`), never "this commit×config×machine already ran → skip."
- `internal/benchruns` is **query/render only** — `Filter` keys on Machine/Model/Precision/Since/Until,
  **no `git_commit`** (`benchruns.go:28,49`).
- nightrun's only skips are time/freshness (`Saturated`, `recheckDays`; `run.go:180,251`,
  `select.go:120-135`) — **not** lineage.
- `benchcli.DetectInvalidation` (`benchcli.go:445`) is the **inverse** (force re-run on drift) and has
  **zero production callers** — the exact freshness predicate reuse needs, sitting dead.

**Verdict → FILED as [#4600](https://github.com/anthony-chaudhary/fak/issues/4600).** Adapt to fak's
idiom rather than copy InferenceX's `/reuse-sweep-run` human gate: at `chooseAction`, look up a prior
run with matching (git_commit × config × machine) lineage, gate it through the currently-dead
`DetectInvalidation`, and emit a new `OutcomeReused` skip when valid. This is benchmark-run-artifact
reuse-to-skip — distinct from agent-reuse.

### Axis 4 — Fused fast-but-wrong gate (ALREADY-SHIPPED — convergence)

**InferenceX mechanism.** The *identical* perf config is re-run for accuracy and gated against a
**per-model threshold resolved most-specific-first** (`generate_sweep_configs.py:173`
`mark_eval_entries` + `validate_scores.py:313` reject if `val < min_score`; `thresholds.json`).
A speed win that corrupts outputs fails the same matrix cell.

**fak witness.** fak converges here: the benchmark-side fused-quality gate
(`internal/deepseekbench`, `CompareSpeedup` refuses a speedup when quality parity is unverified),
and — most tellingly — commit **`5066afd67` (#4540) "set quantization-specific quality budgets"**
plus **#4517** (exact/differential/rubric/statistical oracles) ship exactly InferenceX's
per-artifact, most-specific-first threshold pattern. **Already-shipped; document, don't file.**

### Axis 5 — Sign-off re-verification / witness-not-self-report (ALREADY-SHIPPED — convergence)

**InferenceX mechanism.** A second Claude re-derives the CODEOWNER checklist from evidence
(`codeowner-signoff-verify.yml`); CI independently re-verifies each claimed green run rather than
trusting the checkmark; a green **skipped** aggregator is treated as a false pass.

**fak witness.** This is fak's entire floor: `dos_commit_audit` (claim-vs-diff),
`dos_review` (residual bands), `internal/gateway/subagent_witness.go`, `dos_verify`,
`dos_recall` (re-verify a memory at read time). Deep convergence — the study is a strong external
validation of fak's doctrine, not a source of net-new. **Document, don't file.**

### Axis 6 — Perf-benchmark anti-gaming (DIVERGENCE / distinct surface)

**InferenceX mechanism.** A cluster of *benchmark-integrity* gates: no FLOP-reducing architecture
hacks (`PR_REVIEW_CHECKLIST.md:24`); run the **pinned upstream image as-shipped**, patches only via
a filed `docs/waiver/<PR>.md` (`:29`); a **frozen, non-substitutable golden acceptance-length**
target for spec-decode so systems (not tuned draft-heads) are compared
(`golden_al_distribution/README.md:25,42`); prefix-caching **disabled** on random-token inputs so
caching can't inflate the compute floor.

**fak witness.** fak's anti-gaming checker `internal/antipattern/checker_games.go`
(`SOLUTION_GAMES_CHECKER`, `:15-17`) targets **grader gaming** — an artifact that prints a hardcoded
"pass", a test that short-circuits before its assertion — i.e. *correctness*-benchmark cheating, a
**different surface** from perf-benchmark cheating (FLOP-cuts, cache-friendly workloads, substituted
spec-heads). fak's honest-benchmarking is carried by category-honest ladders + `benchlineagegate` +
the fused-quality gate (Axis 4), not by a "declared unmodified artifact + waiver seam".

**Verdict → divergence, with one latent idea.** The **structured-waiver seam** ("the runtime is a
declared, unmodified artifact; every deviation is logged, not silent") is a clean reproducibility
pattern fak could adopt *if/when* it benchmarks external pinned engines — but fak mostly benchmarks
its own kernels, so it does not map cleanly today. Documented, not filed.

### Axis 7 — Cache-hostile workload (DELIBERATE OPPOSITE CHOICE — worldview contrast)

**InferenceX mechanism.** Every request is a **unique RNG token walk** (`(offsets[i]+i+j) %
vocab_size`, `benchmark_serving.py:319`), dataset locked to `random` (`:999`),
`random-prefix-len 0` by default — deliberately **erasing prefix-cache economics** to measure the
cache-independent per-GPU compute floor for a token-factory buyer.

**fak witness.** fak's worldview is the **opposite and equally deliberate**: prefix-cache reuse
across agent fan-out is a *headline win* fak exists to measure and exploit (the KV-reuse finding;
`internal/cacheobs`, `fak_serving_prefix_cache_hit_rate`). InferenceX sells per-GPU compute floor to
hyperscalers; fak optimizes an agent fleet where intra-group prefix reuse is the actual lever. Both
are right for their audience. **Documented as a validated divergence** — and a reminder that a
borrowed benchmark's *workload design encodes its buyer*; copying InferenceX's random-token workload
into fak would measure the wrong thing.

### Axis 8 — Declaration-triggers-measurement vs auto-adjudicating ratchets (DIVERGENCE)

**InferenceX mechanism.** `perf-changelog.yaml` is an **append-only, byte-immutable ledger**;
appending an entry is the *only* way to get a changed config re-benchmarked
(`process_changelog.py:157-215` + `run-sweep.yml:95-113`), so a **silent perf change is
structurally impossible**. But the measured delta is **report-only** — colorized red/green into the
PR summary (`compare_results.py:17-27`) for a human CODEOWNER. The system **refuses to
auto-adjudicate "is this regression acceptable?"**

**fak witness.** fak's model is the deliberate inverse: scorecard/ratchet gates **auto-block** on
regression. fak *does* automate the accept/reject decision; InferenceX *refuses* to. The
append-only, tamper-evident ledger discipline is a minor latent idea for fak's
`docs/nightrun/*.jsonl` perf ledgers, but the "human judges the delta" philosophy is a genuine
worldview fork worth naming, not adopting. **Documented.**

### Axis 9 — Real power → tokens/MW vs simulated token-per-watt (DIVERGENCE + latent borrow)

**InferenceX mechanism.** A best-effort `nvidia-smi`/`amd-smi` sidecar writes `gpu_metrics.csv`;
aggregation is **load-windowed** and **never breaks the run** (`aggregate_power.py:1-21,260-265`) —
telemetry is optional, result *validity* is not. Per-run it emits **fungible ingredients**
(`tput_per_gpu`, `intvty = 1000/TPOT`, `joules_per_token`, `avg_power_w`); the buyer metrics
(perf-$/tokens-per-MW) are multiplied **downstream** in the dashboard ETL, not in the bench repo.

**fak witness.** fak's token-per-watt is **simulated** (no real watt source on most boxes). The
divergence is structural. **Latent borrow:** the "best-effort, load-windowed power sidecar that
never fails the run" pattern *if* fak wants real power on its real-GPU boxes (the new AMD RX 7600
Vulkan backend could read `amd-smi`). Low priority; documented.

### Axis 10 — Result-validity failure gate (MOSTLY-SHIPPED, verified → narrow residual)

**InferenceX mechanism.** A hard **5% failure-rate gate** invalidates the datapoint
(`benchmark_serving.py:960-968`, `raise SystemExit`); failed requests contribute **0 tokens**
(`:449-450`); a stream that never emits a first token is counted **failed, not zero-latency**
(`backend_request_func.py:311-317`). You cannot buy throughput by dropping/erroring requests.

**fak witness (verified — establish PARTIAL, refuter found the gate already exists).** The claimed gap
is **mostly false**:
- `cmd/loadgen` (self-described "the GPU server benchmark's load matrix") ships a `-max-error-rate` flag
  **defaulting to 0.0** (`main.go:31`) with a hard `if worst > *maxErrorRate { os.Exit(1) }`
  (`main.go:74-81`) — **stricter** than InferenceX's 5%.
- Zero-token exclusion is **universal**: errored requests `continue` before any token add in loadgen
  (`loadgen.go:171-176`) and fold as `Failed++` with no token add in webbench (`serving.go:697-700,710`).
  You **cannot** buy throughput by dropping/erroring requests anywhere.

**Narrow residual (the only real sub-gap):** the webbench **serving-parity** harness counts failures
but does **not** gate on a max-failure-rate — `ValidateParityClaim` / `TrackMeasured`
(`webbench.go:449`, `serving.go:1050-1060`) require only that tracks are "measured" (`Stats.OK>0`), so
a run failing 99% of requests still validates. Minor hardening; #44 (its home) is closed, so **document,
don't file** — mention on any future parity-harness work.

---

## Net-new candidate — FILED

**[#4600 — feat(benchloop): reuse a prior benchmark run by lineage (commit×config×machine) to skip
redundant re-runs](https://github.com/anthony-chaudhary/fak/issues/4600)** — filed 2026-07-13 after the
confirm-the-seam read above (establish ABSENT + adversarial-refute unable to find any reuse-to-skip
path + repo-scoped search returned zero coverage). The issue sites the exact seam
(`benchloop.chooseAction:340`) and the dormant inverse to wire in (`benchcli.DetectInvalidation:445`,
zero callers). Suggested labels left for the maintainer (`enhancement`, `benchmark`,
`track/B-performance`, `class:infra`).

## Ledger of verdicts

| Axis | fak status (post-refute) | Disposition |
|---|---|---|
| 1. SLO-conditioned goodput (joint ttft∧tpot∧e2e) | REAL refinement (has success-rate + single-e2e goodput) | route **#2242** |
| 2. Token-position decode SLO + SLA frontier | ABSENT (verified; fak docs concur) | route **#3079** |
| 3. Benchmark-run reuse by lineage | ABSENT (verified) | **FILED #4600** |
| 4. Fused fast-but-wrong gate | ALREADY-SHIPPED (#4540/#4517, deepseekbench) | convergence |
| 5. Sign-off re-verification / witness | ALREADY-SHIPPED (dos_commit_audit/review/verify) | convergence |
| 6. Perf-benchmark anti-gaming / waiver seam | DIVERGENCE (checker_games = grader-gaming) | document |
| 7. Cache-hostile workload | DELIBERATE OPPOSITE (fak measures reuse) | document |
| 8. Declaration-triggers-measurement ledger | DIVERGENCE (fak auto-ratchets) | document |
| 9. Real power → tokens/MW | DIVERGENCE + latent (amd-smi sidecar) | document |
| 10. 5% result-validity failure gate | MOSTLY-SHIPPED (loadgen `-max-error-rate`=0.0; narrow webbench-parity residual) | document |

## One-line worldview delta

InferenceX benchmarks the **inference system** for a **token-factory buyer** — so it erases caching,
freezes the workload, measures real power per GPU, and **refuses to auto-adjudicate** an acceptable
regression (human reads the delta). fak benchmarks an **agent fleet** for an **operator** — so it
*celebrates* prefix reuse, auto-ratchets regressions, and witnesses every claim. The two share one
floor exactly: **the number must survive a distrustful re-check** — InferenceX via CI re-verifying
each sign-off, fak via `dos_*` witnesses. The borrowable net-new is narrow (SLO-goodput semantics,
position-swept decode, run-reuse); the rest is convergence or a validated fork.

---

## Refresh — pinned InferenceX deep pass (2026-08-24)

This refresh resolves the original study's largest evidence defect: the source is now pinned.
It also covers InferenceX's post-July AgentX and measured-power expansion rather than
re-litigating the ten axes above.

### Source and scope

- **Source:** [`SemiAnalysisAI/InferenceX`](https://github.com/SemiAnalysisAI/InferenceX)
  at `0b0138fd7de0a6f927f9769b19d594d01f586107` (commit date 2026-08-23,
  Apache-2.0). Repository observations below were checked 2026-08-24.
- **Surface read:** root/docs architecture and procedures; workflow graph; matrix/config
  generation; result processing, evaluation and ingestion; AgentX request/backend/power
  adapters; reuse discovery and validation; power integration; representative tests; the
  available 200-commit history; latest release; open/closed issues, merged PRs, and all four
  public discussions.
- **Scale at check:** 1,281 tracked files in the checkout: 510 YAML, 283 shell, 181 JSON,
  179 Python, and 81 Markdown files. This is primarily a continuous benchmark research and
  operations repository, not an inference engine.
- **Method:** four delegated read-only scans were attempted, but InferenceX's own checkout
  hooks entered repeated `PostToolUse Failed` loops. The coordinator stopped them rather
  than trusting partial reports, then independently read the pinned source and public
  GitHub surfaces. No InferenceX code was copied.

### What InferenceX is now

InferenceX treats a benchmark point as a lifecycle, not a timing loop:
configuration matrix -> managed hardware runner -> engine recipe -> client workload ->
raw artifacts -> validated aggregate -> staged/production ingestion. The main extension
unit is a declarative recipe plus narrowly typed processing adapters; CI owns promotion,
recovery, and provenance. The project intentionally compares external serving stacks
(vLLM, SGLang, TensorRT-LLM, Dynamo and TileRT among them), while fak's native-performance
product path must continue to name the fak engine and use those stacks only for explicit
benchmark/parity envelopes.

Three post-July changes materially extend the earlier study:

1. **AgentX is a first-class workload family.** Request aggregation derives throughput,
   latency and multiple interactivity views from timestamped request records
   (`utils/agentic/aggregation/request_metrics.py:137-208@0b0138f`), while backend adapters
   extract service-side capacities and metrics without forcing every engine into one log
   grammar (`utils/agentic/aggregation/backends/base.py:1-79@0b0138f`). Result validation
   refuses missing/invalid required fields rather than converting absence to a zero
   (`utils/agentic/validation/validate_agentic_result.py:20-121@0b0138f`). Recent merged work
   added full-response ITL and normalized interactivity
   ([#2504](https://github.com/SemiAnalysisAI/InferenceX/pull/2504),
   [#2544](https://github.com/SemiAnalysisAI/InferenceX/pull/2544)).
2. **Power became measured, windowed, and scope-versioned.** Single-node and multinode
   paths validate sample timing and integrate device energy over a declared workload
   interval (`utils/aggregate_power.py:161-225@0b0138f`;
   `utils/aggregate_power_multinode.py:541-690@0b0138f`). The AgentX adapter keeps power
   provenance and applicability explicit (`utils/agentic/aggregation/power_adapter.py:18-145@0b0138f`).
   The implementation arrived as an observable four-part sequence—single-node validation,
   multinode validation, official disaggregated collection, then strict monitor lifecycle—
   plus whole-deployment schema semantics
   ([#2323](https://github.com/SemiAnalysisAI/InferenceX/pull/2323),
   [#2437](https://github.com/SemiAnalysisAI/InferenceX/pull/2437),
   [#2456](https://github.com/SemiAnalysisAI/InferenceX/pull/2456),
   [#2494](https://github.com/SemiAnalysisAI/InferenceX/pull/2494),
   [#2599](https://github.com/SemiAnalysisAI/InferenceX/pull/2599)).
3. **Reuse is an operational protocol, not a cache lookup.** Candidate selection binds a
   source PR/commit and workflow run, admits only explicit conclusions, and supports a
   pinned recovery path (`utils/find_reusable_sweep_run.py:40-233@0b0138f`). A separate
   validator checks artifact manifests, expected eval coverage and duplicates before
   ingestion (`utils/validate_reusable_sweep_artifacts.py:113-368@0b0138f`). The history
   shows why each step exists: reuse repeatedly needed recovery and cancellation semantics
   ([#2175](https://github.com/SemiAnalysisAI/InferenceX/pull/2175),
   [#2389](https://github.com/SemiAnalysisAI/InferenceX/pull/2389),
   [#2418](https://github.com/SemiAnalysisAI/InferenceX/pull/2418)).

### Witnessed borrow decisions

| Technique | Current fak witness | Verdict / route |
|---|---|---|
| Measured-window energy receipt with sample coverage and `device` / `role` / `whole_deployment` scope | `internal/metrics/device_spine.go:64,148-151` exposes instantaneous watts, but no benchmark-window joule integral or scope contract | **Net-new adaptation:** [#8773](https://github.com/anthony-chaudhary/fak/issues/8773) |
| Client-phase + backend-log request lifecycle join with source digests and explicit missingness | `internal/agenticbench/latency_phases.go` and `result_packet.go` already own agentic phases/receipts; no backend-adapter lifecycle join was found | **Net-new adaptation:** [#8774](https://github.com/anthony-chaudhary/fak/issues/8774) |
| Quality-constrained full-response interactivity from timestamped stream events | fak has TTFT/ITL and position-SLO work, but no dedicated full-response interactivity receipt; #3079 is adjacent rather than identical | **Net-new adaptation:** [#8775](https://github.com/anthony-chaudhary/fak/issues/8775) |
| Reusable benchmark artifacts with source identity and separate pre-ingest validation | `internal/benchlineagegate/benchlineagegate.go` and `internal/benchloop/reuse.go` now implement the missing lineage/reuse spine; original #4600 is closed | **Convergence:** no new issue |
| Goodput/SLO and position-sensitive latency | Existing #2242 and #3079 remain the right homes | **Duplicate:** no new issue |
| Result-validity gate and performance-change declaration | fak's benchmark authority, claim gates and witnessed commit path already enforce the stronger general contract described above | **Convergence / deliberate divergence:** no new issue |
| InferenceX engine recipes or Python workflow implementation | Product path and language/ownership differ; silent external-engine fallback would violate the native invariant | **Exclude direct port; borrow semantics only** |

### Worldview delta

The July note characterized InferenceX as declaration-triggered continuous serving
measurement. The current tree goes further: **agent behavior is itself a serving workload,
and a publishable point is the joined evidence of client experience, server mechanics,
quality, power, topology, source identity, and ingestion state.** That worldview is useful
to fak, but fak should implement it at its own stronger seam: one native-engine receipt that
can prove task quality, lifecycle, cache/policy effects, and net-true resource cost together.

### Completeness and honest limits

- GitHub surfaces checked on 2026-08-24: 104 total issues, 67 total PRs reported by the
  repository metadata at check time; merged AgentX/power/reuse PR searches; one current
  prerelease (`tilert-v0.1.5.post2-inferencex.1`); four public discussions. These are dated
  observations, not durable counts.
- The checkout was depth-limited to 200 commits, so history older than that window was
  sampled through files and GitHub search rather than exhaustively cloned.
- No production dashboard/database internals or private runner infrastructure were
  available. Public workflows and result contracts prove the handoff, not the private
  service implementation.
- No benchmark number was imported or compared. This study extracts mechanisms only;
  performance claims still require matched, quality-constrained fak-native runs.

**Companions:** original axes in this note; field-borrow routes
[#8773](https://github.com/anthony-chaudhary/fak/issues/8773),
[#8774](https://github.com/anthony-chaudhary/fak/issues/8774), and
[#8775](https://github.com/anthony-chaudhary/fak/issues/8775); refresh tracker
[#8770](https://github.com/anthony-chaudhary/fak/issues/8770).

## Exhaustive inventory refresh (2026-08-25)

Issue [#9003](https://github.com/anthony-chaudhary/fak/issues/9003) refreshed this study against pinned revision [`0b0138fd7de0a6f927f9769b19d594d01f586107`](https://github.com/SemiAnalysisAI/InferenceX/commit/0b0138fd7de0a6f927f9769b19d594d01f586107) (commit timestamp `2026-08-24T03:33:43Z`). The machine-readable denominator is [`docs/research/inventory/semianalysisai-inferencex.json`](../research/inventory/semianalysisai-inferencex.json); its rendered companion is [`docs/research/inventory/semianalysisai-inferencex.md`](../research/inventory/semianalysisai-inferencex.md).

### Complete source-class read-back

- **Tree:** `fak study-inventory` walked all 1,280 non-`.git` files in 11 immediate subsystems: 1,145 runtime files, 34 tests/fixtures, 84 documentation files, and 361,845 text lines. The map pins representative paths for README/docs, architecture/design, runtime source, tests/fixtures, history metadata, and root plus subtree license/provenance.
- **History, releases, and roadmap:** commit history was read through the pinned revision; the sole release was [`tilert-v0.1.5.post2-inferencex.1`](https://github.com/SemiAnalysisAI/InferenceX/releases). No canonical `ROADMAP` file exists, so prospective work was audited across repository TODOs and the open issue/PR surfaces rather than inferred from absence.
- **GitHub denominator at the revision timestamp:** pagination retained records created by the cutoff and found 389 issues (103 open, 286 closed), 2,299 pull requests (52 open, 2,247 closed; 1,520 merged), all four discussions (all open, none answered), and one release. Commands and counts live in the inventory's `non_tree_study.github_surface` and registry `source_evidence`.
- **License/provenance:** the root declares Apache-2.0 in `LICENSE`, `NOTICE`, and `README.md`; experimental/vendor subtrees preserve their own notices and licenses. Borrowing therefore remains idea/pattern level unless a later implementation performs file-level provenance review.
- **FAK self-query:** candidate-specific `fak capabilities` queries covered benchmark accounting, artifact lineage, and agentic lifecycle. They found native benchmark receipts/authority, captured artifacts and module revisions, routing/trajectory/session-control surfaces; no query justified importing an external serving runtime.

### Refreshed candidate decisions

| Candidate | FAK decision | Durable owner / reason |
|---|---|---|
| Scope-versioned measured-power receipts | **Already tracked** | [#8773](https://github.com/anthony-chaudhary/fak/issues/8773) owns workload-window and deployment-scope binding. |
| Client/server lifecycle join for agentic workloads | **Already tracked** | [#8774](https://github.com/anthony-chaudhary/fak/issues/8774) owns cross-boundary phase correlation. |
| Quality-constrained full-response interactivity | **Already tracked** | [#8775](https://github.com/anthony-chaudhary/fak/issues/8775) owns judged-completion latency. |
| InferenceX serving/runtime implementation | **Reject for product path** | Keep as benchmark/reference evidence only; adoption would violate FAK-native ownership of kernels, memory, scheduling, cache, adaptation, and operations. |
| Artifact reuse and ingestion-integrity conventions | **Already owned** | Captured evidence, benchmark authority, and revisioned artifacts cover the reusable pattern; the refreshed denominator found no distinct missing primitive. |

The exhaustive refresh therefore creates **no new follow-on**: all applicable gaps are deduplicated to #8773–#8775. The monitor row records the complete candidate matrix, completeness critic, self-query witness, and issue tracking so a later refresh can compare a pinned denominator instead of repeating an unbounded study.
