# Cactus Needle: borrow the negative tool-call corpus, not the runtime

> Studied 2026-08-28. Source: [cactus-compute/needle](https://github.com/cactus-compute/needle), pinned at [`ee221ce7c13579d9809209b979a9b7a50936614c`](https://github.com/cactus-compute/needle/commit/ee221ce7c13579d9809209b979a9b7a50936614c). Current-tree license: Apache-2.0. Tracker: [#9852](https://github.com/anthony-chaudhary/fak/issues/9852). Durable receipt: `study_ca43928a48a872230ea6bdbfd676249cb26d8d22fd095441b9c792919ad8b49c`.

Companions: [study-repo](../../.claude/skills/study-repo/SKILL.md), [field-borrow](../../.claude/skills/field-borrow/SKILL.md), machine inventory [`cactus-compute-needle.json`](../research/inventory/cactus-compute-needle.json), implementation issue [#9864](https://github.com/anthony-chaudhary/fak/issues/9864).

## Verdict

Needle's best FAK contribution is its **evaluation content**, not its engine. Its compact environment suite tests realistic device actions across six behavioral classes: positive, missing detail, irrelevant tool, explicit negation, invalid input, and parallel calls. FAK already owns the stronger acceptance mechanism—exact-model binding, invalid-tool zero tolerance, per-task HOLD behavior, refusal checks, and parallel-call width—but the frozen Qwen3.8 quant corpus contains only one positive tool fixture. [#9864](https://github.com/anthony-chaudhary/fak/issues/9864) tracks the small, attributed corpus addition.

Most other mechanisms are already present on-axis or do not fit FAK's product boundary. FAK's model descriptor, quantization provenance, KV metadata, and mixed-precision packages already express the artifact contracts Needle embeds in `.cact`. Needle's model architecture and fine-tuning recipes remain WATCH because the repository does not contain the native engine, confidence implementation, retrieval scorer, weights, or matched quality evidence. Its fetched binary, model-confidence authority, and default-on telemetry are explicit EXCLUDE decisions.

No Needle source or artifact was copied in this study. Future native-performance work remains fak-native and Qwen3.8-first; Needle is prior art and a fixture-attribution source, never an automatic engine fallback.

## Value frame

- **For:** maintainers qualifying native Qwen3.8 tool calling before it enters an agent route.
- **Problem:** one successful positive tool call does not reveal over-calling under negation, calls with missing or invalid arguments, attraction to irrelevant tools, or broken parallel binding.
- **Today:** `docs/benchmarks/qwen38-quant/corpus.json` has one `lookup_ticket` tool fixture; `examples/model-acceptance-agentic-v1.json` covers multi-step, parallel, refusal, and retry shapes but not a source-attributed device-action negative corpus.
- **Better because:** adapt the six Needle case classes into FAK's existing fail-closed evaluator, preserving FAK's stronger per-task gate and exact engine/artifact receipts.
- **Witness:** #9864 requires a machine-readable attributed corpus, omission tests for every class, exact call/refusal expectations, and a native Qwen3.8 campaign receipt before any quality claim.

Centrality: **Enabling**. P1 improves the native quality gate without adding prompt/runtime surface; P2 reuses the existing evaluator; P3 keeps the change to a bounded fixture/test slice; P4 makes every class and engine identity checkable.

## What Needle is

Needle is a small-device tool-calling stack split across three boundaries:

1. A Python package builds tool schemas, formats calls, manages one process-global native engine, optionally executes returned Python functions, and exposes extraction/playground paths.
2. JAX source defines training, a custom model, cached decoding, quantization, and export to a `.cact` artifact.
3. A downloaded native library performs constrained decoding, confidence, retrieval, and device inference. That binary is not source in this repository.

That third boundary is decisive. The repository supports source-level claims about its API, training/export contracts, fixtures, and Python-side orchestration. It does not support independent claims about native decoder correctness, kernel performance, confidence calibration, binary provenance, or runtime license.

## Load-bearing mechanisms

### Typed tool schemas and exact target rendering

[`needle/agent/tools.py:18-170`](https://github.com/cactus-compute/needle/blob/ee221ce7c13579d9809209b979a9b7a50936614c/needle/agent/tools.py#L18-L170) maps callable signatures, defaults, docstrings, enums, literals, containers, and Pydantic models to JSON schemas. The practical edge cases matter more than the happy path: unions collapse to their first member, and Python reflection is the product boundary. FAK consumes provider-neutral declared schemas and should not grow a Python-specific reflection SDK in the kernel.

[`needle/model/finetune.py:194-217`](https://github.com/cactus-compute/needle/blob/ee221ce7c13579d9809209b979a9b7a50936614c/needle/model/finetune.py#L194-L217) renders tool schema plus user query, appends reasoning and calls as the target, and masks prompt tokens from loss. Sharing the exact renderer is a sound training recipe, but FAK does not currently own model training. Keep it as a recipe/watch item until a FAK-native training boundary exists.

### Tiny-device negative evaluation

The six domain modules under [`needle/environments`](https://github.com/cactus-compute/needle/tree/ee221ce7c13579d9809209b979a9b7a50936614c/needle/environments) constrain tools to small, closed device operations. Their common harness in [`_harness.py:17-45`](https://github.com/cactus-compute/needle/blob/ee221ce7c13579d9809209b979a9b7a50936614c/needle/environments/_harness.py#L17-L45) checks exact calls, ignores order for parallel calls, optionally maps low confidence to refusal, and requires at least 90 percent overall plus zero critical failures. Missing-detail, negation, and invalid-input cases are critical.

FAK should borrow the **case taxonomy and attributed fixture ideas**, not the aggregate gate or confidence rule. `internal/modelaccept` already holds every mismatched task and has explicit refusal/tool-width contracts. Its stricter behavior remains authoritative.

### Resident identity and runtime limits

[`needle/__init__.py:16-160`](https://github.com/cactus-compute/needle/blob/ee221ce7c13579d9809209b979a9b7a50936614c/needle/__init__.py#L16-L160) locates or fetches a platform binary, binds a ctypes ABI, tracks the process-global engine owner, refuses a base agent on tuned weights, and executes up to eight returned calls. Merged [PR #54](https://github.com/cactus-compute/needle/pull/54) repaired instance clobbering by reinitializing on switches, but switching destroys conversation state; open [issue #94](https://github.com/cactus-compute/needle/issues/94) confirms multiple fine-tunes still cannot coexist.

The identity principle is already PRESENT in `internal/modeldescriptor`: validate model, topology, tokenizer, quantization, storage, backend, kernel, envelope, oracle, and migration identity before allocation. Needle's global engine is negative design knowledge, not a runtime to import.

### Artifact and quantization contracts

[`needle/model/export.py:240-437`](https://github.com/cactus-compute/needle/blob/ee221ce7c13579d9809209b979a9b7a50936614c/needle/model/export.py#L240-L437) emits a self-contained, layer-major, 64-byte-aligned `.cact` file with geometry, KV numerics, mixed weight metadata, pretransposed tensors, and tokenizer data. [`needle/model/quantize.py:301-351`](https://github.com/cactus-compute/needle/blob/ee221ce7c13579d9809209b979a9b7a50936614c/needle/model/quantize.py#L301-L351) applies longest-prefix mixed-bit selection over canonical tensor names.

These effects are already PRESENT in FAK through `internal/modeldescriptor`, `internal/kvquantmeta`, `internal/quantprov`, `internal/mixedprecision`, and the native model precision selectors. The file format is different; the invariant is not. No duplicate issue survives.

Needle also derives a KV window from a fixed 11.5 MiB budget and clamps/rounds it in [`architecture.py:598-616`](https://github.com/cactus-compute/needle/blob/ee221ce7c13579d9809209b979a9b7a50936614c/needle/model/architecture.py#L598-L616). FAK's `internal/kvbudget` deliberately adapts to hardware and workload. The tiny-device constant is a worldview signal to WATCH, not a broad default.

### Model and fine-tuning architecture

The visible model combines lexical Engram memory ([`architecture.py:146-206`](https://github.com/cactus-compute/needle/blob/ee221ce7c13579d9809209b979a9b7a50936614c/needle/model/architecture.py#L146-L206)), mHC residual routing, gated grouped-query attention, and Walsh-Hadamard MLP transforms ([`architecture.py:305-426`](https://github.com/cactus-compute/needle/blob/ee221ce7c13579d9809209b979a9b7a50936614c/needle/model/architecture.py#L305-L426)). Sparse attention LoRA keeps factors and merge deltas in fp32 and skips near-zero groups ([`finetune.py:254-433`](https://github.com/cactus-compute/needle/blob/ee221ce7c13579d9809209b979a9b7a50936614c/needle/model/finetune.py#L254-L433)).

Those are research candidates, not deployment evidence. Cached-decoder parity, incremental Engram parity, mHC scan behavior, KV masks, mixed CQ, and tokenizer-reference behavior lack direct equivalence tests. `_doc_prefix_len()` in `decode.py` returns zero. Open [issue #91](https://github.com/cactus-compute/needle/issues/91) reports fine-tuned decision drift across 2-bit and 4-bit artifacts, while follow-up identifies a train/inference template mismatch as an alternative cause. Open [issue #25](https://github.com/cactus-compute/needle/issues/25) questions out-of-box quality. Any kernel proposal would need SOTA review and matched Qwen3.8 evidence first.

## Source and history ledger

Observed around `2026-08-28T21:45:00Z`.

| Source | State | What it changes |
|---|---|---|
| Current tree | [`ee221ce7`](https://github.com/cactus-compute/needle/tree/ee221ce7c13579d9809209b979a9b7a50936614c), 283 commits | Pins the source conclusions; the checkout was clean and 58 tracked files were mapped. |
| Tags/releases | ten lightweight tags through [`v2.0.10`](https://github.com/cactus-compute/needle/commit/7bd8a635ebb977489703b3bb916c21412e8a9cdf), zero GitHub Releases | Current main is six commits ahead; tag names are weak shipped evidence because package versions drift and slow tests are outside the release gate. |
| Package metadata | [`pyproject.toml:1-25`](https://github.com/cactus-compute/needle/blob/ee221ce7c13579d9809209b979a9b7a50936614c/pyproject.toml#L1-L25) | Current source declares version `2.0.8` despite tag `v2.0.10`; runtime dependencies were narrowed at the pin. |
| Forge | 43 issues: 16 open/27 closed; 55 PRs: 10 open/28 merged/17 closed-unmerged | Activity is high, but review evidence is sparse; open proposals are not accepted design or quality proof. |
| Evaluation proposal | open [PR #42](https://github.com/cactus-compute/needle/pull/42) | Proposes multiset parallel-call scoring, BFCL-style buckets, false-trigger rates, and training corrections; useful emerging evidence, not shipped behavior. |
| KV proposal | open [PR #44](https://github.com/cactus-compute/needle/pull/44) | Proposes fixed per-layer KV and device-side decode, but targets an older architecture and retains host automaton state. |
| Confidence/router evaluation | [issue #61](https://github.com/cactus-compute/needle/issues/61) | Reports English-heavy calibration, worse Spanish tokenization, stale confidence after LoRA, and a false-positive ordering that monotonic recalibration could not repair. |
| License transition | [`b188e110`](https://github.com/cactus-compute/needle/commit/b188e110bcee93736ae8eca48dc1982264a6d41d) | Current main changed from MIT to Apache-2.0 on 2026-08-17; historical and open-PR heads require license-at-their-own-revision review. |
| Telemetry | merged [PR #96](https://github.com/cactus-compute/needle/pull/96), [`_telemetry.py:12-84`](https://github.com/cactus-compute/needle/blob/ee221ce7c13579d9809209b979a9b7a50936614c/needle/_telemetry.py#L12-L84) | Anonymous telemetry is default-on with opt-outs; FAK excludes this implicit egress posture. |

There is no changelog, GitHub Release, discussion board, milestone, tracked roadmap, NOTICE, submodule, or vendored third-party tree. The release workflow tags HEAD after a workspace-only version mutation and runs a fast gate, so tags do not prove model/export coverage.

## FAK on-axis witness

Candidate-specific `fak capabilities` and `fak dev index` queries were paired with raw source search.

- Model artifact identity, topology, tokenizer, quantization, KV metadata, engine and migration contracts are PRESENT in `internal/modeldescriptor`, `internal/kvquantmeta`, and `internal/quantprov`.
- Deterministic mixed precision is PRESENT in `internal/mixedprecision`, `internal/model/mixed_precision.go`, `internal/model/exl2.go`, and the per-tensor admission surfaces.
- Exact-model, refusal, retry, invalid-tool, and parallel-width acceptance mechanisms are PRESENT in `internal/modelaccept`; every mismatch yields HOLD rather than relying on an aggregate score.
- Hardware-aware KV budgeting is PRESENT in `internal/kvbudget`; adopting the fixed Needle budget would be divergent outside its tiny-device envelope.
- No on-point capability or corpus was found for a six-class, domain-varied tiny-device negative tool-call suite. The frozen Qwen3.8 corpus has one positive tool call. This is the sole filed gap.
- No capability was found for trusting model-emitted confidence or loading an unpinned native library, and both absences are intentional security/provenance properties.

## Candidate matrix

| Borrow/axis | FAK verdict | Portfolio/license | Decision |
|---|---|---|---|
| Six-class tiny-device tool-call corpus | **PARTIAL** | **DEFAULT · ADAPT** | File [#9864](https://github.com/anthony-chaudhary/fak/issues/9864); preserve FAK's stricter per-task gate and source attribution. |
| Python callable schema reflection | **ABSENT** | **EXCLUDE · INSPIRE-ONLY** | Provider-neutral schemas stay the kernel boundary; union loss is a disconfirming edge case. |
| Artifact-bound geometry/KV/quant/tokenizer identity | **PRESENT** | **DEFAULT** | Keep FAK's engine-neutral descriptor/provenance split; no duplicate issue. |
| Deterministic mixed tensor precision | **PRESENT** | **DEFAULT** | Keep FAK's generic selectors and provenance; no Needle format dependency. |
| Fixed-byte tiny-device KV window | **PARTIAL** | **WATCH · INSPIRE-ONLY** | Retain hardware/workload-aware FAK budgeting; reconsider only for an explicit device class. |
| Shared train/runtime renderer and target-only loss | **ABSENT** | **RECIPE · INSPIRE-ONLY** | Training-boundary recipe, not a kernel default. |
| fp32 sparse attention LoRA | **ABSENT** | **WATCH · INSPIRE-ONLY** | Needs template-parity, quantized-behavior, and confidence-recalibration evidence. |
| mHC + Engram + gated GQA + Hadamard model | **ABSENT** | **WATCH · INSPIRE-ONLY** | Qwen3.8-first native goal and missing matched quality evidence block promotion. |
| Model confidence as execution/security authority | **ABSENT** | **EXCLUDE** | Opaque, stale after LoRA, language-sensitive, and incompatible with the capability floor. |
| Unpinned fetched native library | **ABSENT** | **EXCLUDE** | FAK requires digest/revision/engine receipts and remains fak-native. |
| Default-on anonymous telemetry | **ABSENT** | **EXCLUDE** | Egress must be explicit and capability-governed. |

## Negative knowledge

- A constrained decoder can still emit invalid JSON; closed [issue #85](https://github.com/cactus-compute/needle/issues/85) was not accompanied by a linked captured regression in its thread.
- Confidence calibration is not a policy boundary. It is English-heavy, can become stale after LoRA, and cannot repair a wrong score ordering with a monotonic threshold.
- Process-global engine ownership is not multi-model residency. Reloading repairs clobbering but discards state and serializes identities.
- Quantized behavior drift must separate quantization from prompt-template parity; issue #91 currently admits both causes.
- Tag presence is not release integrity when version mutations are not in the tag tree and slow correctness tests are excluded.
- A self-describing artifact helps compatibility but does not replace cross-decoder parity, malformed-artifact, tokenizer-reference, or conversion-quality tests.
- The default telemetry opt-out and unpinned binary fetch are precisely the kinds of convenience defaults FAK's capability/provenance floor should refuse.
- Historical experiments—FFN Matryoshka, saliency selection, residual attention, noising reconstruction, contrastive training, and audio augmentation—are not current Needle 2 capabilities merely because their commits remain in history.

## License and provenance

The pinned current tree is Apache-2.0 and is compatible with adaptation when the license and attribution are preserved and modified copied files are marked. This study copies no code or assets. The repository used MIT before [`b188e110`](https://github.com/cactus-compute/needle/commit/b188e110bcee93736ae8eca48dc1982264a6d41d), so historical/open-PR source must be evaluated at its own head.

No NOTICE, CLA, per-file SPDX header, submodule, or vendored provenance ledger was found. The license changed after outside contributions without a public relicensing-consent record found in this audit. Prefer current-tree ideas and original FAK implementations; seek legal/maintainer confirmation before substantial historical copying. Native binaries, Hugging Face artifacts, the separate Cactus runtime, and assets were not licensed by this repository study.

## Completeness and limits

The source fan-out mapped all 58 tracked files across runtime/API, schemas, environments, data/fine-tuning, architecture/decode, quantization, export, telemetry, packaging/release, tests, and license. It inspected all 283 commits and 10 tags at history level, all 43 issue and 55 PR states/conversations at forge level, and deep-read material implementation, evaluation, release, license, telemetry, quality, and portability threads. Focused runtime tests passed: 45 passed and one native-engine test skipped.

The coordinator independently re-read each worker result. Fresh local source/test evidence confirmed the runtime and model claims; fresh GitHub API/history/license evidence confirmed the history claims. Worker self-reports were not used as completion evidence.

Unavailable or deliberately out of scope: GitHub Projects V2 (token lacks `read:project`), every open PR's full diff/test execution, Actions logs, PyPI artifacts, Hugging Face weights/tokenizer/native library, the separate `cactus` runtime, `needle-rs`, and binary asset authorship. These omissions block runtime, performance, release-integrity, binary-license, and broad portability claims. They do not affect the source-visible corpus gap or the exclusions above.

Refresh when main moves, a new tag/Release appears, PR #42/#44 lands or changes, issues #25/#61/#91/#94 materially resolve, the native engine becomes source-addressable, or FAK's Qwen3.8 acceptance corpus ships #9864.

## Filed trail

- [#9852](https://github.com/anthony-chaudhary/fak/issues/9852): study tracker.
- [#9864](https://github.com/anthony-chaudhary/fak/issues/9864): the sole surviving PARTIAL/ABSENT on-axis borrow—an attributed six-class tiny-device tool-call corpus for FAK's existing Qwen3.8 acceptance path.
- No issue was filed for PRESENT, WATCH, RECIPE-only, DIVERGENT, or EXCLUDE rows.
