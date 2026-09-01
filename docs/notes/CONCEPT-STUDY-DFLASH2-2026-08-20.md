---
title: "DFlash 2 study: parallel path selection, suffix decay, and the FAK measurement seam"
description: "Pinned study of DFlash 2's candidate selector, dynamic local convolution, exact verification path, reported H200 results, runtime support, evidence limits, and FAK disposition."
---

# DFlash 2: parallel path selection, suffix decay, and the FAK measurement seam

## Verdict

DFlash 2 is a credible improvement to **draft quality**, not a new verification
algorithm. It keeps the expensive drafter forward pass parallel, then adds two cheap
forms of local dependence: a top-16 candidate lattice whose final path is selected by a
low-rank predecessor/current compatibility score, and two-tap dynamic grouped
convolutions around every attention and MLP sublayer. The target-model verification
step remains the correctness boundary.

The published numbers are promising but remain upstream-reported. On one H200, the two
released cards report batch-1 speedups of 2.67-3.43x for Qwen3.8-27B and 3.08-4.62x for
Muse Glimmer 30B. At concurrency 32 those ranges fall to 1.01-1.45x and 1.15-1.68x.
There is no DFlash 2 paper, training recipe, raw per-request result ledger, uncertainty
analysis, or independent reproduction in the studied source set.

FAK should **ride**, not reimplement, the trained DFlash2 architecture today. SGLang
merged it after its latest release; vLLM, llama.cpp, and Ollama support are still open
PRs; oMLX ships one fork release. The immediate FAK-sized borrow is model-independent:
record accepted and proposed counts by draft position so the reported suffix decay is
visible instead of collapsed into one mean. That spine is filed as
[#8258](https://github.com/anthony-chaudhary/fak/issues/8258). The study also corrected
a stale closure: stochastic rejection-sampling losslessness remains absent in FAK, so
existing [#4202](https://github.com/anthony-chaudhary/fak/issues/4202) is reopened.

## Observation identity

| Field | Witnessed value |
|---|---|
| Observed at | `2026-08-20T12:38:55-07:00` |
| Article | [DFlash 2: Keep Drafting Parallel](https://inco.ai/blog/dflash2/), published 2026-08-18 |
| Captured article | 287,468 bytes; SHA-256 `4683b60e0c3fa588a7deaae5641cd61d5ddad5844983db945f8b1c1fc345b912` |
| Reference repository | [`z-lab/dflash`](https://github.com/z-lab/dflash) |
| Pinned reference revision | [`07ebd93db9f472af339b644bb70221ad8428328a`](https://github.com/z-lab/dflash/commit/07ebd93db9f472af339b644bb70221ad8428328a), tag/release `v0.1.0` |
| Repository signal | 5,809 stars, 413 forks; discovery signal only |
| Qwen card pin | [`incoai/Qwen3.8-27B-DFlash2@dedf8df`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/tree/dedf8df68adfb1afeaf7b7480c0a0243108177b4) |
| Muse card pin | [`incoai/Muse-Glimmer-30B-DFlash2@8336acb`](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2/tree/8336acb8dc9b8bf9c25f12d7785ee6df26703119) |
| FAK comparison state | `internal/model@r447+g21aa6f1e4f`, `internal/polymodel@r12+g25ef19f068`, `internal/deepseekbench@r2+gf354231c79` |

Refresh this note when SGLang releases a tag containing PR #35371, any of vLLM
#52816 / llama.cpp #27342 / Ollama #17865 merges or closes, DFlash 2 publishes a paper
or training/evaluation ledger, or FAK ships #8258 / #4202 / a native draft source.

## Feynman-simple value frame

- **For:** operators whose target model is expensive enough that one verification pass
  can amortize several accepted output tokens.
- **Problem:** fully parallel block drafters make cheap proposals, but later positions
  lose coherence because each position has weak information about its predecessor.
- **Today:** DFlash 1 keeps drafting parallel but commits an independently chosen token
  at every position; FAK can verify/accept drafts but cannot see acceptance decay by
  position and has no compatible trained native drafter.
- **Better because:** DFlash 2 injects predecessor dependence after the parallel logits
  exist and adds a tiny local receptive field inside the drafter, raising accepted
  tokens without making the main drafter pass autoregressive.
- **Witness:** immutable model cards report acceptance length and output tok/s against
  autoregressive, MTP/DFlash, and DSpark arms; FAK's first borrow is a deterministic
  accepted/proposed position profile, not an inherited speedup claim.

## Problem centrality and P1-P4

**Enabling.** DFlash 2 can improve the decode engine below FAK's kernel checkpoint, but
it does not replace managed context, policy adjudication, routing, or witness semantics.

- **P1 managed context:** the drafter consumes verified context and target hidden-state
  taps; no new prompt-management mechanism transfers to FAK.
- **P2 net-true efficiency:** acceptance length must be joined with drafter cost, target
  verification cost, concurrency, memory, and output tok/s. Acceptance alone is not a
  gain.
- **P3 bounded adaptation:** candidate sampling remains subject to exact target
  verification. Any later FAK depth controller must only change work, never output.
- **P4 integrated operations:** model/runtime revision, sampling policy, proposal depth,
  position profile, quality parity, latency, and hardware must share one result ledger.

## What DFlash 2 actually changes

### 1. Keep the drafter's expensive pass parallel

Base DFlash predicts a block of masked positions in one forward pass over target hidden
features. The DFlash 2 checkpoints remain five-layer non-causal/sliding-attention draft
models. Their configs declare a block size of 8 for Qwen and 16 for Muse, with target
hidden-state taps, not an autoregressive draft loop.

The architecture is therefore unlike EAGLE/DSpark-style sequential draft generation:
the learned dependence is added around or after the parallel hidden/logit computation.

### 2. Select one coherent path through top-k candidates

For every draft position, the reference implementation keeps the top `K` target-head
logits (`K=16` in both released checkpoints). It projects the current hidden state to
rank 256, multiplies it with a predecessor embedding, scores each candidate through a
successor embedding, adds the candidate's unary logit, then greedily or stochastically
walks one path
([`dflash/model.py:515-547@07ebd93`](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py#L515-L547)).

The walk is sequential across a small `K x K` lattice, but the expensive model and
vocabulary projections are already complete. The blog reports this selector adds about
2M parameters and 0.6% draft-cycle latency on Qwen3-4B while increasing GSM8K mean
acceptance length from 4.27 to 4.61 at temperature 0 and 3.78 to 4.25 at temperature 1.

### 3. Add two-tap dynamic grouped convolution inside each draft layer

Each attention and MLP sublayer gets a prepare/finish wrapper. A learned projection
produces position-dependent grouped coefficients; a base kernel plus those coefficients
mixes the current and prior block positions before and after the sublayer
([`dflash/model.py:423-512@07ebd93`](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py#L423-L512),
[`dflash/model.py:639-650@07ebd93`](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py#L639-L650)).

The released configs pin kernel size 2 and group size 16. The blog reports +16.5M
parameters and +0.7% cycle latency for this component, motivated by a measured decline
in draft recall toward the block suffix. This is local, block-scoped recurrence; it does
not create a persistent cross-request state machine.

### 4. Keep target verification as the losslessness boundary

At greedy decoding, verification accepts only the target-argmax prefix and emits the
target bonus token. At sampling temperatures, the reference path computes the selector's
sparse draft distribution, accepts with the usual target/draft probability rule, and on
rejection samples the normalized positive residual
([`dflash/model.py:94-124@07ebd93`](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py#L94-L124),
[`dflash/model.py:251-292@07ebd93`](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py#L251-L292)).

The selector can improve acceptance, but it cannot certify its own output. Exactness
comes from this verifier. That distinction is load-bearing for FAK: draft quality and
lossless acceptance are separate capabilities and separate issues.

## Reported results and their limits

The immutable model cards say both evaluations used SGLang on one NVIDIA H200,
FlashAttention 3 for target and draft attention, maximum 4,096 new tokens, and the five
tasks GSM8K, MATH-500, HumanEval, MBPP, and MT-Bench.

| Target / block | Baseline mean acceptance | DFlash 2 mean acceptance | Batch-1 DFlash 2 speedup | Concurrency-32 speedup |
|---|---:|---:|---:|---:|
| Qwen3.8-27B / 8 | MTP 4.28; DSpark 3.62 | 4.80 | 2.67-3.43x | 1.01-1.45x |
| Muse Glimmer 30B / 16 | DFlash 4.44; DSpark 4.48 | 5.70 | 3.08-4.62x | 1.15-1.68x |

These are upstream-reported measurements, not FAK observations. The cards provide
aggregate tables, configuration prose, and benchmark-format pointers, but no raw
request/result rows, exact SGLang revision, environment/container lock, random seeds,
run counts, error bars, energy/cost accounting, or output-quality artifact. The
reference repo has a runnable benchmark client but no committed DFlash 2 result bundle
and no training code. The article's component ablations are likewise summary tables.

The concurrency curves matter: the speculative advantage compresses as the target
becomes better utilized. On Qwen at concurrency 32, MTP and DSpark are slower than
autoregressive on several tasks and DFlash 2 reaches only 1.01x on MT-Bench. A FAK
adoption claim therefore needs the target operating regime, not a batch-1 headline.

## Runtime state at the observation date

| Runtime | Pinned state | Disposition |
|---|---|---|
| SGLang | PR [#35371](https://github.com/sgl-project/sglang/pull/35371) merged as [`c14312a66`](https://github.com/sgl-project/sglang/commit/c14312a66420b75ca9a11bf1817c4db1fa26b097), followed by quantized-head fix `1cf2b8c54`; observed main `61fa64ae7` | SHIPPED on main, but no release newer than `v0.5.17` contains it. Recipe may pin main; package-release docs must wait. |
| vLLM | PR [#52816](https://github.com/vllm-project/vllm/pull/52816) head `66e5414c6`, 885 added/changed implementation/test lines | WATCH: open, conflicting/needs-rebase at observation. Do not call it released support. |
| llama.cpp | PR [#27342](https://github.com/ggml-org/llama.cpp/pull/27342) head `5ecbe1ac17`, 20 files / 676 additions | WATCH: open. It includes GGUF metadata, graph, lattice walk, and sampling changes but no PR-specific test file. |
| Ollama | PR [#17865](https://github.com/ollama/ollama/pull/17865) head `1c7808a61c`, 17 files / 635 additions | WATCH: open; MLX-specific integration with unit tests. |
| oMLX fork | release [`0.6.2-dflash2`](https://github.com/z-lab/omlx-fork/releases), tag `46225aebee`; signed arm64 DMG digest `sha256:94f56e...d5e5bd21` | SHIPPED fork artifact for Apple Silicon; not evidence of upstream oMLX support or cross-platform parity. |

SGLang and vLLM both carry real regression tests around candidate selection. That makes
the implementations stronger evidence than the blog diagram, but it does not reproduce
the published benchmark tables on the pinned hardware.

## Negative knowledge

- There is no distinct DFlash 2 paper at the observation date. The blog cites the base
  DFlash, Domino, DSpark, Canon Layers, Dynamic Short Convolutions, and ConvLLM papers as
  antecedents; none is a substitute for a DFlash 2 methods/evaluation artifact.
- `z-lab/dflash@07ebd93` ships inference models, an MLX implementation, and a benchmark
  client. It does not ship the DFlash 2 training pipeline or a committed raw benchmark
  ledger.
- The model cards say greedy output matches and sampling preserves the target
  distribution. The source supports that mechanism, but this study did not run the
  27B/30B checkpoints on H200 hardware or reproduce distributional tests.
- Parameter and cycle-latency overhead are upstream microbenchmarks. FAK has no observed
  memory-residency, load-time, page-in, or billed-cost result for these checkpoints.
- Candidate selection and local convolution are coupled to trained weights. Copying the
  inference architecture without training/evaluation evidence would create machinery,
  not a useful FAK drafter.

## Worldview

DFlash 2 assumes agent workloads make decode volume a first-order bottleneck and that
parallel proposal remains the right constraint. It prefers small learned compatibility
and local-recurrence modules over extra full transformer layers or sequential draft
passes. It also treats verification as the immutable semantics layer: the drafter may be
aggressively optimized because rejection only changes work, not output.

That worldview mostly agrees with FAK's deny-by-witness design. The tension is product
placement: FAK should own correctness, routing, measurement, and ride-mode integration;
specialized serving engines should own trained GPU/Metal draft architectures until FAK
has a native draft source and an observed reason to bring the machinery in-kernel.

## FAK inward map

FAK self-query, indexed docs/leaves/verbs/claims, raw source search, and open-issue
read-back establish the exact axis split:

- `internal/model@r447+g21aa6f1e4f` has a live `SpecDecodeGreedy` binding to
  `VerifyForward`; target verification, accepted-prefix commit, and rollback are
  **PRESENT for the CPU greedy envelope**.
- `internal/polymodel@r12+g25ef19f068` returns scalar accepted totals and mean acceptance
  length. `DraftLengthThrottle` keeps a rolling scalar accepted/proposed ratio.
  Accepted/proposed counts by draft position are **ABSENT**; #8258 owns the spine.
- `docs/industry-scorecard/decoding.md` correctly fences output-distribution parity to
  greedy decoding. No stochastic rejection/residual accept path exists in current
  `internal/model` or `internal/polymodel`; #4202 was reopened after its prior closure
  was traced to a docs-only commit.
- [#3197](https://github.com/anthony-chaudhary/fak/issues/3197) owns a native EAGLE-style
  draft source, [#5154](https://github.com/anthony-chaudhary/fak/issues/5154) owns a
  DeepSeek V4 MTP head, and [#5261](https://github.com/anthony-chaudhary/fak/issues/5261)
  owns a model-free n-gram drafter. DFlash2 is an alternative trained source, not a
  reason to duplicate those prerequisite seams.
- FAK's vLLM/SGLang ride path already delegates speculative decoding to the serving
  engine. A DFlash2 recipe is configuration/documentation after a stable upstream
  release, not a new FAK runtime architecture.

## Candidate matrix refresh

### Candidate-by-candidate disposition

| Technique | Exact axis | FAK status | Portfolio route | Disposition |
|---|---|---|---|---|
| Target verify + greedy prefix accept/rollback | output identity at T=0 | PRESENT in bounded CPU envelope | DEFAULT | Keep; no issue. |
| Rejection sampling against sparse selector probabilities | output distribution at T>0 | ABSENT | DEFAULT correctness | Existing #4202 reopened; no duplicate. |
| Accepted/proposed counters by draft position | suffix-decay observability | ABSENT | DEFAULT | Filed #8258; model/GPU independent. |
| Automatic depth from the position curve | workload adaptation | PARTIAL: scalar throttle only | WATCH | Consider only after #8258 supplies a stable signal and a replay proves net gain. |
| Top-k low-rank candidate path selector | proposal coherence | ABSENT | OPTIONAL-MODULE / WATCH | Keep behind a native trained draft-source spine and same-checkpoint ablation; no issue now. |
| Two-tap dynamic grouped convolution | local draft dependence | ABSENT | OPTIONAL-MODULE / WATCH | Keep with the trained DFlash2 architecture; copying it without training weights is not useful. |
| SGLang-main DFlash2 ride | external serving | PRESENT as generic ride substrate | RECIPE | Add a versioned recipe only after a release contains #35371. |
| vLLM / llama.cpp / Ollama ride | external serving | upstream-unreleased | WATCH | Refresh on PR merge/close; no FAK implementation issue. |
| oMLX fork ride | Apple Silicon external serving | external artifact exists | RECIPE / MANUAL | Operator may use the signed fork; do not imply upstream or non-Apple support. |
| Import published speedup as a FAK claim | net-true efficiency | ABSENT evidence | REJECT | Requires a FAK-observed workload/hardware ledger with quality parity and all work counted. |

No selector/convolution issue is filed because the smallest working native spine is still
a compatible trained drafter, already owned by the draft-source portfolio. The dated
promotion trigger is a stable checkpoint/runtime pair plus an apples-to-apples replay
showing the selector or convolution adds net output tok/s after memory and verifier cost.

## Licensing and provenance

The reference DFlash code is MIT-licensed. SGLang, vLLM, and the oMLX fork are
Apache-2.0; llama.cpp and Ollama are MIT. The two model cards declare Apache-2.0. A
stdlib-only Go adaptation of a merged mechanism can therefore be **ADAPT/INSPIRE** with
the applicable notices; open-PR code remains inspire-only until merged or separately
cleared.

The blog itself supplies no software reuse license, so its prose/figures are
**INSPIRE-only**. Model-card metadata does not by itself settle every target-model,
training-data, or weight-redistribution right. FAK should not vendor these checkpoints
or article assets; operators can name external model/runtime revisions. This is a
good-faith engineering review, not legal advice.

## Source ledger

| Source class | Exact source | State/date | What changed the conclusion |
|---|---|---|---|
| Article | `https://inco.ai/blog/dflash2/`, captured SHA-256 above | published 2026-08-18 | Established thesis, component ablations, reported overhead, and runtime links. |
| Reference implementation | `dflash/model.py`, `benchmark.py`, README, packaging, license at `07ebd93` | release `v0.1.0`, 2026-08-18 | Confirmed selector formula, convolution placement, stochastic verification, benchmark client, and absent trainer/result ledger. |
| Model cards/configs | Qwen pin `dedf8df`; Muse pin `8336acb` | committed 2026-08-19 | Fixed block/K/rank/kernel parameters, hardware/sampling prose, and all reported task/concurrency tables. |
| SGLang implementation/history | PR #35371 merge `c14312a66`; follow-up `1cf2b8c54`; observed main `61fa64ae7` | merged 2026-08-19; unreleased at check | Proved one production runtime has merged code/tests and exposed an immediate quantized-head follow-up. |
| Other runtime implementations | vLLM #52816 `66e5414c6`; llama.cpp #27342 `5ecbe1ac17`; Ollama #17865 `1c7808a61c`; oMLX `46225aebee` | three open PRs; one fork release | Prevented conflating announced integrations with merged/released portability. |
| Prior papers/code | DFlash arXiv:2602.06036v2; Domino 2605.29707v1; DSpark 2607.18413v1; Canon Layers 2512.17351v2; Dynamic Short Convs 2606.03825v1; ConvLLM 2607.05147v1 | versions captured 2026-08-20 | Supplied antecedents only; none closes the missing DFlash2 paper/training/evaluation artifact. |
| FAK self-query | capabilities; docs/leaf/verb/claim indexes; raw `rg`; #23/#3197/#4202/#5154/#5261/#8258 | FAK HEAD `35dbbcf2` at comparison | Split present greedy verification, stale stochastic gap, absent position profile, existing drafter portfolio, and ride path. |

## Delegated-read witness partition

Three read-only workers were assigned model/paper, Python-runtime, and native-runtime
source classes. The model/paper and Python-runtime turns died on a harness parameter
error before a terminal report; the native-runtime turn was interrupted after failing to
return within the study window. Their narration was not folded.

- **CONFIRMED delegated results: 0/3 terminal result groups.**
- **DEAD: 2/3** (`prompt_cache_retention` unsupported by the selected model).
- **UNWITNESSED/interrupted: 1/3.**
- **Independent replacement witness:** article hash/read, fresh Git checkouts, immutable
  revisions, GitHub PR/release API read-back, model-card/config read-back, source diffs,
  license files, and live FAK source/issue read-back. Every claim used above comes from
  that non-agent-authored evidence.

## Completeness critic

Read the full article; inventoried all tracked files and read the README, reference
PyTorch/MLX model paths, generation/verification loop, benchmark client, packaging,
license, release tag, and available history in `z-lab/dflash`. Read both full model cards
and configs. Acquired all six cited paper source bundles and their linked code where
published; used them as antecedent checks rather than attributing DFlash2 results to
them. Read the SGLang implementation diff, unit-test files, merge/follow-up history and
release list; the full vLLM, llama.cpp, and Ollama PR file lists plus their mechanism,
configuration, sampling, and test seams; and the oMLX integration/release history and
artifact metadata. Queried every relevant PR/release state fresh on 2026-08-20.

On the FAK side, queried capabilities and indexes, then read the actual greedy binding,
verify path, chain/tree run results, scalar throttle, scorecard, serving adapter docs,
prior DFlash/spec-decode/DeepSpec notes, and all adjacent open/closed issues. Raw search
was used as a false-absence check for DFlash2, position profiles, path selectors,
convolution drafters, and stochastic acceptance.

Not run: the 27B/30B checkpoints, H200/SGLang benchmarks, Apple signed binary, or open-PR
GPU/Metal tests. Those require large external weights and hardware and would still not
recover the absent upstream raw result/training ledger. No community Slack/Discord or
private issue context was accessed. These limits are carried into the verdict rather
than treated as passes.

No material technique remains unclassified. Immediate work is #8258 and reopened
#4202; selector/convolution work is explicitly WATCH-gated; runtime integrations are
versioned ride recipes, not silently deferred native implementations.

## Companions

- [Base DFlash study](CONCEPT-STUDY-DFLASH-2026-07-10.md)
- [Speculative-decoding mechanism study](CONCEPT-STUDY-SPECDECODE-2026-07-10.md)
- [DeepSpec borrow study](deepspec-borrow-study-2026-07-11.md)
- [vLLM study](CONCEPT-STUDY-VLLM-2026-07-18.md)
- [SGLang study](CONCEPT-STUDY-SGLANG-2026-07-18.md)
- [llama.cpp study](CONCEPT-STUDY-LLAMACPP-2026-07-18.md)
- [#8258: per-position speculative acceptance profile](https://github.com/anthony-chaudhary/fak/issues/8258)
- [#4202: stochastic lossless acceptance](https://github.com/anthony-chaudhary/fak/issues/4202)

## Exhaustive inventory refresh (issue #8994, 2026-08-25)

The denominator is now explicit in
[`docs/research/inventory/z-lab-dflash.json`](../research/inventory/z-lab-dflash.json).
It was generated with `fak study-inventory` from a detached checkout of
`07ebd93db9f472af339b644bb70221ad8428328a`: **11 regular files** across **14 tree entries**, **3 immediate
subsystems**, and only `.git` skipped. The pinned tree is small enough that every
file was read or classified, not sampled.

### Complete source-class read-back

| Source class | Pinned or dated evidence | Result |
|---|---|---|
| README/docs | `README.md` at `07ebd93` | Product claim, architecture figure, install/use, model matrix, benchmark table, backend links, and citations are all in one README; there is no docs tree. |
| Architecture/design | `README.md`, `dflash/model.py`, `dflash/model_mlx.py` at `07ebd93` | Design is embedded in the figure and paired Torch/MLX implementations; no separate design document exists. |
| Runtime source | all five `dflash/*.py` files, `pyproject.toml`, and both workflows at `07ebd93` | Reference package contains Torch and MLX model paths, CLI, benchmark harness, package metadata, and publishing automation. |
| Tests/fixtures | exhaustive 11-file/14-entry map at `07ebd93` | Confirmed absent: no tests, fixtures, test workflow, or declared test runner. This limits the reference package's regression value. |
| History/changelog/releases | complete pinned git log; GitHub tag/release API on 2026-08-25 | One release and tag, `v0.1.0`, both at `07ebd93`; no changelog. The 24-commit history was read, including the DFlash 2, MLX, and serving-link additions. |
| Open/closed issues, PRs, discussions | GitHub GraphQL and complete `gh ... --limit 200` state reads on 2026-08-25 | 85 open + 48 closed issues; 11 open + 8 closed + 3 merged PRs; 7 discussions. Threads cover correctness/load failures, backend integrations, training requests, smaller models, and non-CUDA support. These are dated mutable forge facts, not revision-pinned tree facts. |
| Roadmap/TODOs | all source plus issues, PRs, discussions, and milestones | No canonical roadmap, milestones, or source TODOs. Requests are distributed across forge threads and must not be presented as maintainer commitments. |
| License/provenance | `LICENSE`, `pyproject.toml`, README paper/model links at `07ebd93` | MIT covers the repository code; model checkpoints and papers remain separately linked upstream artifacts whose terms must be checked independently. |
| FAK self-query | `fak capabilities` plus `fak-dev index docs/leaves` queries recorded in the registry row | FAK already has the study and shipped per-position acceptance telemetry (#8258); stochastic lossless sampling remains open (#4202). No fak-native DFlash architecture implementation surfaced. |
| Candidate matrix | [candidate dispositions](#candidate-by-candidate-disposition) | Eight candidates retain explicit adopt/adapt/reject/defer decisions; this refresh changes evidence completeness, not those technical decisions. |
| Completeness critic | inventory map plus this table | Every required class is backed by a pinned path, a dated forge query, or an explicit checked-absent result. |
| Issue tracking | FAK #8258, #4202, parent #8936, child #8994 | No new implementation issue is justified by an inventory-only refresh; existing shipped/open owners remain the correct follow-ons. |

### FAK decisions after denominator refresh

- **Keep / shipped:** per-position acceptance telemetry remains the transferable
  runtime primitive; #8258 is closed and independently visible in FAK's index.
- **Keep / open:** #4202 remains the exact correctness owner before any sampled
  DFlash-like path can claim losslessness at `T > 0`.
- **Do not import:** the tiny reference package has no tests and is not a
  production serving backend. FAK should not vendor it or silently route native
  inference through linked vLLM/SGLang integrations.
- **Defer:** trained DFlash layers, checkpoints, and training recipes remain
  architecture/model work, not an inference-only kernel patch. Revisit only with
  a fak-native model path, quality gate, and matched end-to-end witness.
- **Borrow when owned:** retain path coherence, dynamic local mixing, exact target
  verification, and per-position telemetry as design constraints for any future
  native implementation.

### Follow-ons and completeness critic

No new follow-on was filed: the only actionable FAK gaps found by the exhaustive
pass are already owned by shipped #8258 and open #4202. Upstream's issue volume,
backend requests, and absent tests are useful risk evidence, but do not establish
new FAK scope. The principal residual uncertainty is external to this inventory:
model/checkpoint licensing and performance/quality must be re-verified whenever a
specific artifact and runtime envelope are selected. The registry's row-specific
`fak study-monitor --inventory-check --json` result is the readiness witness for
this refresh.
