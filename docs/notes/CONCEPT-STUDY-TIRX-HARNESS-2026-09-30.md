# TIRx Harness study — 2026-09-30

Status: **deep source study, draft recommendations; no integration or performance qualification**.
Observed at 2026-09-30T05:41:36Z, with subsequent same-session source checks. The operator requested read-only research and a local deliverable. Consequently no issues, commits, registry updates, persistent study-store additions, deployments, installs, remote GPU requests, or user-code uploads were performed. This is an explicit scope override of `study-repo`'s default filing/landing workflow, not a claim that its full filed-study lifecycle completed.

## Actual local skill and acquisition

The repository discovery adapter `.agents/skills/study-repo/SKILL.md` points to the canonical `.claude/skills/study-repo/SKILL.md`. Both were read; the canonical `field-borrow` witness instructions were also read. The executor/cloud catalogs and personal Codex skill directory did not reveal a competing personal study skill. This pass used the maintained repository workflow, not an invented summary workflow.

Public source checkouts were acquired into temporary scratch; exact operator paths are deliberately omitted here. No upstream instructions were treated as authorization to execute software.

| Source | Pinned revision | Source event | State / relevance |
|---|---|---|---|
| [TIRx Harness](https://github.com/mlc-ai/TIRx-harness) | `ccd7a04a72aa121d98fd90b2a4a3fe97ca5a6726` (H) | 2026-09-29T19:13:39Z | Main includes generated API docs; runtime source inspected |
| [TIRx kernels](https://github.com/mlc-ai/TIRx-kernels) | `909d8e0b9889dc49ac69abf4fda7a88a48389def` (K) | 2026-09-29T16:06:00Z | Curated kernels, DSL, tests and benchmark contracts |
| [tvm-rust-ext](https://github.com/mlc-ai/tvm-rust-ext) | `516fdfdd1117091937cf55928c21dfeea310cf28` | Harness gitlink | Initialized only for source/provenance reading; Apache LICENSE/NOTICE inspected |
| [Launch post](https://blog.mlc.ai/2026/09/29/tirx-harness-an-open-compiler-harness-for-agentic-gpu-programming) | Publication snapshot | 2026-09-29T18:30:00Z | Motivation and reported Blackwell evaluation; mutable prose |
| [Documentation](https://tirxharness.mlc.ai/docs/) | Repository `docs/` at H | Same source revision | Local source supplied documentation when initial web fetch failed |
| fak | `68fa650adcd137b19b3a06e05c5b36884055fc01` | Local snapshot | Clean tree at acquisition; comparison seams read locally |

Every H/K `path:line` below denotes code at the full SHA above, not mutable main. Refresh after compiler/DSL releases, checker coverage changes, benchmark promotions, relevant PR merges, or a new target device. `fak study search --limit 20 TIRx` returned `[]`; `fak-dev study-monitor --due-days 14` worked and no TIRx ledger entry was located. No persistence claim is made for this draft.

## What it actually is

TIRx Harness supplies a **compiler-facing environment for agents developing kernels**. It does not serve models or replace fak's inference engine. `evolution/setup.py:142-186@H` prepares environments, a workspace, prompt and manifest, then delegates actual agent execution to external Claude, Codex or Humanize. No native context-compaction engine, inference scheduler, complete trajectory store, or general permission mediator was found in this layer. Frontier persistence is predominantly a prompt-prescribed file/Git protocol, not an enforced admission service.

The motivation is concrete: make compiler behavior predictable enough that agents can connect edits to instructions, turn intermittent correctness failures into useful diagnostics, reuse concrete implementations, and separate GPU measurement from concurrent agent activity. This is reconstructed from the launch post, workload specs and actual code; it is a design rationale, not proof that every claimed outcome holds.

### Compiler and authoring boundary

TIRx-lite is an opinionated PTX-level subset of TVM's TIRx IR. It exposes typed addresses, explicit launch/warp roles, shared-memory allocation/swizzles, barriers, phase cursors and instruction sequences; it excludes general high-level tile/layout abstractions. It is not a new whole-model runtime.

The implementation chain is [Python tracing and PrimFunc construction](https://github.com/mlc-ai/TIRx-kernels/blob/909d8e0b9889dc49ac69abf4fda7a88a48389def/tirx_kernels/tirx_lite/entry.py#L546) → `IRModule` → [`tvm.compile(..., tir_pipeline="tirx")`](https://github.com/mlc-ai/TIRx-kernels/blob/909d8e0b9889dc49ac69abf4fda7a88a48389def/tirx_kernels/tirx_lite/entry.py#L319) → CUDA. Default targets use exact CUDA architecture names; an environment variable can retarget prepared runs. Source spans and low-level IR guards preserve useful diagnostic structure (`entry.py:546-622@K`). Void PTX calls must be explicitly emitted during tracing or bare expressions disappear (`tirx_lite/__init__.py:88-95@K`). `pipeline.py:15-100@K` implements stage/phase and full/empty barrier protocols using CUDA-specific classes, TMA transaction counts and tcgen05 commit operations.

The harness does not contain the whole compiler. It depends on `apache-tvm==0.27.0`, FFI `0.1.14.post0`, CUTLASS DSL `4.7.0`, NVRTC `13.2.78`, nvdisasm `13.3.73`, separate `tirx-kernels` and KCoral packages (`pyproject.toml:27-47@H`). Documented setup is Linux x86_64/Python 3.12–3.13; released wheels also include Linux aarch64. Darwin library filename handling in `setup.py` is not evidence of supported Mac installation or Metal lowering. No native Metal/ROCm/Vulkan implementation was established in the studied tools/DSL/kernel sources.

### CPU analysis: useful, bounded evidence

NumSim transpiles supported concrete TIRx instructions into cached native Rust CPU artifacts. It provides deterministic numerical feedback with explicit operation coverage; it needs an independent numerical reference and cannot establish GPU timing, absence of races, or arbitrary numerical correctness. Clocks, SM IDs and some hardware choices use deterministic representatives. Unsupported forms reject rather than silently reverting to another backend (`numsim/engine-rs/SUPPORTED_OPS.md`, `numsim/transpiler/build.py:56-62,180-240@H`).

Synccheck examines interleavings **holding each warp's traced path and values fixed**. Racecheck uses byte ranges, per-lane/asynchronous vector clocks and proxy fences for traced accesses. Neither enumerates all alternate inputs, ordinary-memory values, atomic return orders and resulting control-flow paths. The [public runner](https://github.com/mlc-ai/TIRx-harness/blob/ccd7a04a72aa121d98fd90b2a4a3fe97ca5a6726/tirx_harness/src/tirx_harness/numsim/checker_runner.py#L706) currently requires exactly one kernel phase. Reports distinguish `clean`, `review`, `incomplete`, and `error`; `.require_clean()` rejects all other verdicts. A clean report is invocation-scoped and still needs independent GPU correctness.

The corpus admits real limitations: several TMEM ordering cases retain REVIEW despite numerical success (`tests/numsim/corpus/canonical_verdict_rationale.md:17-35@H`). That honesty is a useful mechanism to borrow. CPU tests and same-kernel GPU parity fixtures exist; their presence is source evidence only. No third-party tests were run in this study.

### Search, contracts and measurement

Tasks bind computation, shapes/dtypes, candidate interface, independent oracle, tolerances, GPU and measurement boundary. For example, `fp16_gemm_floor.yaml:12-29@H` specifies 12 B200 shapes, setup outside timing, current input consumption/output overwrite, and a 99%-element numerical match at declared tolerances. That floor is workload-specific, not a model-quality guarantee.

`evolution/prompts/PROMPT.md:17-42,80-105@H` retains reproducibly passing, self-contained candidates from materially different mechanisms even when slower than the current winner. This prevents a single fastest family from erasing useful search directions. It is prompt guidance; the coordinator still needs an enforced evidence gate.

Repeated correctness checks poison outputs, synchronize, and check after timing; exact repeatability is optional (`benchmark_common.py:44-89,690-755,977-1040@H`). KDA additionally mutates sequence metadata within the same pointer to expose stale identity-keyed schedule caches (`kda_forward_portfolio_multishape.py:4220-4248@K`). That is particularly relevant to fak's resident state/cache ownership.

Two acceptance traps were independently confirmed by reading implementation:

- [FP16 aggregation](https://github.com/mlc-ai/TIRx-harness/blob/ccd7a04a72aa121d98fd90b2a4a3fe97ca5a6726/evolution/benchmark/flashinfer_bench_evolve/tasks/fp16_gemm_floor/benchmark.py#L83) excludes failed rows, and `adapter.py:272-290@H` discards returned rows. A zero exit or printed speedup can coexist with incomplete correctness. Require every expected row ID and PASS verdict.
- [Timing](https://github.com/mlc-ai/TIRx-harness/blob/ccd7a04a72aa121d98fd90b2a4a3fe97ca5a6726/evolution/benchmark/flashinfer_bench_evolve/benchmark_common.py#L264) prefers CUPTI median with cold L2/no CUDA graphs but catches any exception and falls back to averaged CUDA events. Record actual timer mode/cache conditions and prohibit unmatched comparison.

The kernel collection has stronger outer contracts: exact architecture/reference identity gates, fresh subprocess correctness tests, and an outer requirement for one pass/zero failures/zero skips (`registry.py:167-184,253-282@K`, `tests/test_correctness.py:137-169@K`). Paired benchmark comparison rejects dirty/mismatched provenance and failed/polluted rows (`bench_suite/ratio_diff.py:337-450@K`). These are different surfaces; do not generalize the FP16 adapter trap to every suite.

### Remote execution and originality controls

KCoral separates client development from GPU execution. The kernel suite uploads source; the server's compiler/dependencies are what run, so local TVM edits do not become measured compiler changes (`bench_suite/README.md:43-61@K`). The server implementation/deployment was not audited here; scheduling, authentication, TLS and isolation claims need their own witness.

The [generic remote runner](https://github.com/mlc-ai/TIRx-harness/blob/ccd7a04a72aa121d98fd90b2a4a3fe97ca5a6726/evolution/remote/kcoral_remote.py#L7) explicitly says restrictions are **not sandbox enforced**: `--cmd` executes Bash as the server user with machine filesystem access. Dedicated adapters reject traversal/symlinks and constrain artifacts, but uploaded Python still executes in a worker and inherits environment. This is not equivalent to fak's capability floor or hostile-code containment.

Sanitized worktrees, installed packages and recreated history address a different concern: preventing agents from recovering forbidden seed kernels through Git history (`preparation/worktree.py:29-43,205-257@H`). Shell hooks are heuristic controls. Evaluation originality protection and operating-system security are separate axes.

## Performance and maturity: what the evidence supports

The [launch post](https://blog.mlc.ai/2026/09/29/tirx-harness-an-open-compiler-harness-for-agentic-gpu-programming) reports **2.94× KDA forward versus FlashKDA and 6.84× backward versus FLA**, family geometric means on evaluated Blackwell workloads. It explicitly measures **GPU kernel time, not end-to-end application latency**. It used Humanize 2, web disabled, with reference versions from the September 25–26 sweep. These are upstream-reported results, not reproduced fak results or causal ablations of the harness components.

The pinned collection includes wins and losses. `bench_suite/baseline.md:385-387@K` lists FP16/BF16 GEMM ratios .979/.893/.998 and `:473-475` NVFP4 .836/.952/.933 (ref/ours, so below one is slower). Selected KDA backward rows are much faster. Large square GEMMs do not qualify batch-one skinny decode; attention tests do not qualify paged KV or serving; quantization time excluded from NVFP4 timing must reappear if relevant to an inference claim.

The baseline header records 232 successful rows, an opaque timestamp `4`, compiler `7a8c0703` and kernel revision `52d04aed`; current README describes 274 rows/93 device kernels. Roster breadth and performance coverage differ. This discrepancy prevents treating every current source row as performance-qualified; it does not disprove the separate launch evaluation.

Public repository history is young: 11 harness commits and 6 kernel commits at the pins, with substantial code imported in initial commits. Harness v0.1.2 was released September 29, 2026; main includes later docs changes. [Harness PR 4](https://github.com/mlc-ai/TIRx-harness/pull/4) reports real B200 validation of docs/examples but discloses unresolved local Humanize CUDA Error 304 and no 24-hour optimization/1.1× target witness. [PR 10](https://github.com/mlc-ai/TIRx-harness/pull/10) is still open for bundling skills into wheels; do not call that shipped. [Kernel PR 3](https://github.com/mlc-ai/TIRx-kernels/pull/3) is open for typed tensor-map CUDA-host export and depends on Apache TVM PR 20489; do not count Go/C host-bundle interoperability as already complete.

Open and closed upstream issues/PRs, release metadata, and selected review-comment endpoints were read. The issue lists mostly contain PRs, not an established field failure corpus. Harness discussions are disabled; no separate substantive roadmap/RFC surfaced. Dependency-wide TVM/KCoral history was not mined.

## License/provenance gate

At H, no root LICENSE/NOTICE or project/Cargo license declaration was found for the harness's own code. GitHub metadata also reports null license. Its packaging copies Apache notices for `tvm-rust-ext`; that dependency grant does **not** license the harness. **Harness expressive source/tests/prompts are INSPIRE-ONLY pending an explicit grant.** No source was copied into fak implementation.

K has Apache-2.0 root terms and file-specific combined Apache/MIT/BSD terms plus NOTICE/licenses. FA4 explicitly carries Apache AND BSD and frozen upstream origin. cuDNN, FlashInfer, DeepEP and SGLang notices must travel with copied ports. Compatible reuse may be ADAPT/DIRECT-PORT after a selected-file audit; do not blanket-license all code from the repository name.

## Candidate matrix: compare at the axis

Queries used `fak-dev feature query` for `kernel compiler`, `race synchronization`, `hardware benchmark`, plus capabilities/index and varied raw source searches. Catalog results were discovery hints, followed by code inspection. No broad capability hit was treated as proof of a narrow axis. All dispositions are snapshot/proposal assessments, not runtime parity claims.

| Borrow / specific axis | Source anchor | fak seam / on-axis assessment | Disposition and first falsifier |
|---|---|---|---|
| Source-correlated low-level kernel IR | `entry.py:546-622@K` | `internal/compute/strix/compiler_toolchain.go:137` has Clang path, but default dry-run and predefined metadata; **PARTIAL** | WATCH/ADAPT DSL lessons. Generate one real kernel artifact and prove diagnostic source mapping; reject synthetic metadata as silicon evidence. |
| Intra-kernel synchronization interleavings | `sync_fixed_verifier.rs:214-230@H` | `vulkan_batch_hazards.go:174` covers inter-dispatch declared ranges, not thread-level traces; **ABSENT in searched compute seam** | WATCH/INSPIRE-ONLY. One barrier-generation bug must be diagnosed before GPU and still reproduced on silicon. |
| Per-lane async/proxy memory races | `race_check_python.rs:62-82,490-547@H` | Same Vulkan hazard seam; **ABSENT in searched seam** | WATCH/INSPIRE-ONLY. One missing proxy-fence example plus clean control; keep invocation scope explicit. |
| Independent numerical kernel oracle | `numsim/api.py@H`, corpus rationale | `internal/hil/microdose_darwin.go:229` repeats the same kernel; `computetune/tune.go:177` already accepts independent reference; **PARTIAL**, not globally absent | DEFAULT enhancement proposal. Make an intentionally repeatable wrong kernel fail reference comparison; preserve physical HIL. |
| Explicit analysis coverage and four verdicts | `checker_runner.py:706-735@H` | HIL pass/fail is physical probe evidence, not semantic coverage; **PARTIAL on checker scope** | WATCH/INSPIRE-ONLY. Unsupported operation must produce incomplete and never promote. |
| Architecture/shape/reference admission | `registry.py:167-184,253-282@K` | `cuda_arch.txt:1`, `cudaarch.go:14`, `computetune/tune.go:139` cover matrix/exact identity; **PARTIAL on kernel-level contract** | OPTIONAL-MODULE/ADAPT. Reject one unsupported exact arch before compilation and one missing reference before correctness acceptance. |
| Repeated poison/overwrite/post-timing checks | `benchmark_common.py:690-755@H` | `computetune/tune.go:177` oracle-before-measure; **PARTIAL on after-timing/stale-output axis** | DEFAULT enhancement/INSPIRE-ONLY. One stale-output candidate must fail after inputs change. |
| Same-pointer metadata mutation witness | `kda_forward_portfolio_multishape.py:4220-4248@K` | `computetrace/trace.go:39` input digests present; **PARTIAL**, cache-specific seam still needs focused audit | RECIPE/ADAPT. Change sequence lengths without pointer change and compare a fresh oracle; drop if current seam already checks content. |
| Diverse self-contained optimization frontier | `PROMPT.md:80-105@H` | `computetune` selects winners; distinct-family persistence not established there; **PARTIAL** | WATCH/INSPIRE-ONLY. Retain two justified families and recover the alternative after objective change; reject archive sprawl. |
| Matched source/device/timer/quality receipts | `ratio_diff.py:337-450@K` | `internal/nativeperf/receipt.go:31,169` and `gate.go:110` bind native matched controls; **PRESENT on engine/quality promotion**, narrower timer details require review | Adapt useful detail into existing receipt, not duplicate framework. Changed timer/cache/input must invalidate comparison. |
| Paired/swapped-buffer sanity controls | `bench_suite/README.md` ratio-trust section | Native matched receipts exist, placement-swap control not established; **PARTIAL** | RECIPE/ADAPT. Byte-identical kernels under swapped buffers should reveal placement effects, not report an algorithm win. |
| Banned-reference isolation including history | `worktree.py:29-43,205-257@H` | Product capability security serves a different goal; **DIVERGENT for production**, useful for search evaluation | RECIPE/INSPIRE-ONLY. Demonstrate forbidden seed cannot be recovered from prepared history; never substitute hooks for OS containment. |
| Remote centralized GPU timing | `bench_suite/README.md:43-61@K` | fak fleet/HIL is the owned execution boundary; **DIVERGENT as a third-party service default** | WATCH optional self-hosted experiment after server review. Verify GPU exclusivity/compiler identity; this study authorizes no service use. |
| NVFP4/FA4/KDA concrete kernel techniques | `nvfp4_gemm.py:26-65,992-1034@K`; FA4/KDA modules | Existing CUDA/Metal compute leaves; **PARTIAL/needs op-specific SOTA witness** | OPTIONAL-MODULE/WATCH. Single measured bottleneck, exact ABI/shape/quality envelope; product benefit must survive native whole-model measurement. |

Worldview finding: kernel optimization benefits from **concrete successful implementations plus ISA references**, because summaries can discard transferable dataflow detail. Source fact: curated zoo plus PTX knowledge; inferred principle: retain executable examples near diagnosis; fak opportunity: small provenance-pinned kernel exemplars tied to real bottlenecks; disconfirming check: examples increase incorrect reuse or fail to reduce time to verified task completion. This is a roadmap consideration, not an automatically approved new framework.

## Concrete recommendations, ordered

1. **Strengthen one existing kernel witness first.** Use the independent-reference seam in `computetune` to add changed-input/output-poison/post-timing qualification for one native kernel. On Metal, clarify that repeatability HIL is not an independent oracle. Acceptance: a deterministic wrong/stale-output kernel fails; the actual native kernel passes its oracle and physical probe. No TIRx installation is required to apply this lesson.
2. **Evaluate trace-level diagnostics as a separate compiler research leaf.** Resolve harness licensing first. Select a single CUDA barrier/proxy hazard, fixed concrete input and supported representation; compare CPU diagnostics with an independent GPU reproduction. Test coverage/incomplete semantics before expanding. Cost and semantics make this research, not an immediate default compiler replacement.
3. **If a measured native CUDA bottleneck warrants it, pilot one K-licensed kernel technique.** Keep model loading, residency, scheduling, KV and launch control in fak. Pin source/compiler/arch/input/quantization ABI; compare one variable with exact timer/cache settings and held-out correctness. Kernel-time evidence then needs matched `engine=fak-native` TTFT/decode/task-completion evidence before promotion. Prefer Qwen3.8 as repository policy requires; upstream Qwen3-Next/KDA names do not prove model compatibility.

Priority alignment: recommendation 1 enables tier 1 all-in-one and tier 2 serving; 2 is enabling compiler research; 3 is a bounded tier 2 candidate with tier 1 qualification. For local native users / repeated wrong outputs or opaque kernel failures / today existing probes and tuning cover some axes / better because independent truth and actionable diagnostics shorten verification / witness the deliberately failing control and real native path. Do not adopt Python/TVM dependencies into the single Go binary by default, import the remote shell runner, replace the inference engine, or claim Metal performance from CUDA evidence.

Bounded duplicate lookup found no fak issue matching `TIRx`; a broader all-state `kernel race synchronization` search returned unrelated/mixed work and is not a full issue admission check. No recommendations were filed. Filing would require fresh scoped dedupe/contracts and current source witnesses.

## Coverage, verification and remaining limits

Four parallel readers covered (1) compiler/NumSim/sync/race/Rust/checker corpus, (2) evolution/workload/remote/guards/tests, (3) DSL/kernel registry/selected implementations/bench/provenance, and (4) actual fak seams. Coordinator independently read the compile path, one-phase rejection, remote-runner warning, FP16 aggregate, timer fallback, baseline headers, HIL repeatability, tuning/reference gate and native receipt checks.

Completeness critic: all load-bearing harness/DSL/benchmark/security boundaries were sampled deeply, with source classes inspected or named. Not every generated operation/test or every one of the many ported kernels was line-read; selected GEMM/NVFP4/FA4/KDA representatives establish mechanisms, not exhaustive kernel parity. Fetched third-party wiki implementations and the entire Apache TVM/compiler history were not acquired; their terms/behavior are not independently audited. KCoral server/deployment and Humanize internals were not studied, so scheduler/containment/trace claims stop at adapter boundaries. No live GPU run, integration, installation or performance reproduction occurred. These are limits of this read-only study, not local-hardware blockers.

Repository source/test presence and upstream PR validation are distinguished from tests personally executed. Unresolved harness licensing blocks source copying; it does not block independent fak witness improvements. CUDA-host portability and skill-wheel packaging remain proposed at observation time. Recheck them at PR merge/release.

The deliverable is a new draft note only. Existing INDEX/monitor registry and persistent study receipt store were left untouched to preserve other landing work. Therefore no `study_...` ID, indexed registration, issue filing or witnessed shipped commit is claimed.

Companions:
- [Actual study workflow](../../.claude/skills/study-repo/SKILL.md)
- [Field-borrow witness discipline](../../.claude/skills/field-borrow/SKILL.md)
- [Local-agent milestone](../local-agent-milestone.md)
- [Native inference ownership and matched envelopes](../native-inference-goal.md)

Next checkable step: choose one existing fak-native kernel and demonstrate that its qualification rejects a repeatable wrong output before proposing any compiler integration.
