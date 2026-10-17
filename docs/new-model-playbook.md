---
title: "Adding a new model to fak — pinned intake, fast path, and scaffold path"
description: "A maintainer playbook for compiling a pinned model release into a refusal-safe onboarding packet, then recognizing and validating the architecture in fak's in-kernel engine."
---

# Adding a new model to fak

This page is the procedure for making fak's **in-kernel reference engine** recognize and run a
new model architecture. It does **not** cover fronting a model through the gateway — any model your
upstream serves is already fronted unchanged (see [supported models](supported/models.md), Layer 1).
This is Layer 2: the pure-Go forward pass.

Start every release with the deterministic intake below. **Pretrained-drafter integration is a P0
onboarding item for every new model**, including aliases and checkpoints with bundled draft heads.
Only an accepted packet reaches the architecture decision; refusal is a result, not a prompt to
guess or fall back.

## Compile the pinned release first

```console
fak new-model --from-manifest release.json --json
```

The read-only compiler accepts `fak.new-model-manifest/1` JSON from local disk and emits a
`fak.new-model-onboarding-packet/1`. It performs no network request, model import, weight allocation,
code write, runtime selection, or scaffold application. The synthetic, non-claiming Qwen3.8-family
contract fixture is
[`internal/newmodel/testdata/qwen38-valid.json`](../internal/newmodel/testdata/qwen38-valid.json).
Its format-valid digest strings are deliberately synthetic and prove only compiler/refusal behavior;
they are not artifact provenance, model-support evidence, or benchmark inputs. A real intake uses
`release.evidence_class: "pinned-release"` and witnessed digests; fixtures use
`"synthetic-non-claiming"`.

The manifest pins four identities before any support work begins:

- the source URI and immutable source revision;
- the source-manifest SHA-256;
- the exact model artifact SHA-256; and
- the tokenizer, chat-template, and context-configuration SHA-256 values.

The manifest also requires an explicit rollback action. Because intake is read-only, the action
normally says to discard the packet and retain the previous native descriptor; later consumers must
carry that action forward rather than inventing an implicit rollback after state changes.

It also declares architecture aliases, the dimensions consumed by `internal/modeldescriptor`, a
closed list of semantic deltas, the still-open semantic/oracle/backend/test/docs/performance
obligations, and the candidate's coupling counts and budget. The compiler normalizes and sorts those
set-like fields, validates the resulting `modeldescriptor.Descriptor`, and emits its digest and
existing `modeldescriptor.Check` report. Recompiling the same logical manifest therefore produces
byte-identical JSON.

The packet always names `engine: "fak-native"` and
`external_runtime_fallback: false`. Compatibility evidence from Transformers, vLLM, SGLang, or
llama.cpp can inform an obligation, benchmark, or oracle; it cannot become an automatic execution
path or close a fak-native rung.

## P0: inventory the pretrained drafter on every intake

Every accepted packet now includes `pretrained_drafter` with schema
`fak.new-model-pretrained-drafter/1`, `priority: "P0"`, and the exact target release,
artifact digest, and tokenizer digest. This is an additive output of the existing
`fak.new-model-manifest/1` compiler; existing manifests need no new input field.
The compiler owns seven `pretrained-drafter-*` obligation IDs and adds them to the
same `obligations`, `support_ladder`, and `registration_closure.open` surfaces as
all other model work. A manifest cannot replace these IDs or mark them complete.
Accepted replay rows carry the same generated obligations and semantic gaps; refused
rows retain their input obligations. Legacy `--family` scaffold output begins with
a P0 prerequisite directing callers to this pinned intake; a scaffold is preparation
and cannot bypass it. There is no separate completion tracker. Scope owner:
[#9628](https://github.com/anthony-chaudhary/fak/issues/9628); visibility coordinates
with [#9630](https://github.com/anthony-chaudhary/fak/issues/9630).

The intake is read-only: `artifact_availability: "not-surveyed"`,
`implementation: "unassessed"`, `software_verification: "not-established"`,
`hardware_qualification: "not-established"`, `serving_activation: "not-admitted"`,
`execution_admitted: false`, and `default_enabled: false`. These are the evidence
limits of compilation, not a claim that a particular model has no existing code.
Descriptor readiness labels, another model's receipts, published weights, a study,
a ticket, or a generated scaffold cannot promote this inventory. The original
manifest digest continues to identify the input; packet bytes and replay outcome
digests also depend on the compiler revision.

### Mandatory evidence checklist

Each check emits its stable obligation ID, kind, `pending` status, description,
and machine-readable `evidence_required` fields. Address these before optional
performance polishing; start with available trained artifacts rather than
mistaking an algorithm scaffold for a compatible pretrained drafter.

| Obligation suffix | Required evidence |
|---|---|
| `artifact-availability-and-pins` | Survey official sidecars **and bundled tensor namespaces**. Record source revision, manifest and every shard digest, tokenizer/template identity, license, and locally verified bytes. If none is published, preserve a dated, source-pinned search and explicit target-only disposition; do not invent an artifact or implicitly authorize training. |
| `model-tokenizer-tap-quant-compatibility` | Exact training-target revision when available, actual loaded target identity, vocabulary/ID mapping, shared embedding/head contract, ordered hidden-state tap indices and pre/post-block convention, independent target/draft quantization, and conversion provenance. An unknown training pin stays unknown. Model names or equal hidden width do not establish compatibility. |
| `tensor-inventory-and-loader` | Names, shapes, dtypes, scale layouts, payload bounds, missing/extra-tensor refusal, exact loader entry point, and a real checkpoint-header witness. A source-derived shape table is not a header-verified loader. |
| `native-forward-and-state-cache` | Reachable fak-native trained forward, target-feature capture, state identity, bounded draft/cache memory, snapshot/restore/trim behavior, and an independent forward oracle. Metadata recognition or a generic speculative loop is insufficient. |
| `exact-verification-rollback-stop` | Greedy target-token parity; exact sampling-distribution proof or explicit unsupported refusal; anchor/proposal/bonus accounting; rejection at every position; rollback; EOS/stop; cancellation and safe session reuse. |
| `serving-activation-and-defaults` | Normal serve/agent entry-point execution, emitted executed-route receipt, qualified model/device envelope, kill-switch drill, unsupported refusal, zero foreign-runtime fallback, and the actual default policy. A flag default or callable helper is not serving activation. |
| `measured-end-to-end-benefit` | Accepted tokens per round **with proposed-token and round denominators**, acceptance by position, draft cost and verify cost separately, quality parity, end-to-end throughput, TTFT/latency, repeated trials, startup/recovery/memory costs, and exact source/artifact/hardware/workload pins against matched target-only execution. |

### Keep status dimensions separate

Use the existing [product/evidence/queue axes](progress-state-defaults.md) on the
owning issue. For its drafter evidence, distinguish:

- **Artifact availability:** unreviewed, published sidecar, bundled weights, or
  not found by a dated pinned search. Published metadata is distinct from local
  digest/header verification.
- **Preparation:** study, ticket, and scaffold are individually named work
  artifacts. None means a trained forward is implemented.
- **Implementation:** identify the landed native loader/forward/state/serving
  slice and its commit. A loader-only slice earns only loader credit.
- **Tested:** name the passing test and exact source/artifact envelope; tests
  authored but not run remain unverified. Synthetic tests do not prove real
  checkpoint execution.
- **Hardware-qualified:** retain an actual physical execution receipt for the
  claimed backend, model/quantization, workload, quality, and inclusive costs.
  Upstream numbers and another drafter's receipt cannot supply this status.
- **Activation/default:** record disabled, opt-in, or qualified default-on only
  from the exercised serving route and qualification policy.

The survey and applicability assessment are always P0. An evidence-backed
`no-artifact` or `not-applicable` disposition (for example, a model that does not
perform autoregressive token generation) belongs against these same obligation IDs
on the owning issue. It earns no implemented/tested/hardware-qualified credit and
must not be inferred from a readiness label. There is no manifest-supplied skip flag.

Qualification applies to a specific trained drafter and target combination.
Keep the target-only path's independent support status; a P0 draft gap neither
licenses a foreign engine fallback nor turns target-only evidence into speculative
performance credit. Intake packets remain immutable; reconcile progress and
receipts on the existing owning issues, then regenerate the inventory when needed.

### Known P0 gaps: source snapshot, 2026-10-17

This dated inventory was checked against
`internal/model@77774015fdb5f42957508dce6a2759ab1de2a983` and the pinned upstream
sources below. It records the baseline before subsequent loader candidates;
re-read those files before promoting a row. No hardware performance or default-on
claim is made by this checklist.

#### Qwen3.8-27B + DFlash2

- **Published weights:**
  [`z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4),
  Apache-2.0. `model.safetensors` is published as 3,848,817,896 bytes with SHA-256
  `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.
  This published digest has not been recomputed from local weights by this inventory;
  actual header names, dtypes, offsets, and non-overlap remain to be witnessed.
- **Compatibility:** the stock five-layer draft uses block size 8 (one anchor plus
  at most seven proposals), hidden width 5120, vocabulary 248320, mask ID 248070,
  shared target embeddings/head, and ordered zero-based **post-block** target taps
  `[5, 19, 33, 47, 61]` from a 64-layer target. Reference math:
  [`dflash/model_mlx.py@07ebd93db9f472af339b644bb70221ad8428328a`](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model_mlx.py)
  (MIT). The observed
  [`Qwen/Qwen3.8-27B` config revision](https://huggingface.co/Qwen/Qwen3.8-27B/blob/72a217afab8029b39e4af1c7273a829995a3dbaf/config.json)
  does not pin the actual loaded target's 18 shards, tokenizer, template, or the
  training-target revision; those remain unresolved. Target quantization does not
  automatically qualify the draft's precision or shared projections.
- **Study/tickets/scaffold:** the [DFlash2 study](notes/CONCEPT-STUDY-DFLASH2-2026-08-20.md)
  and [#12541](https://github.com/anthony-chaudhary/fak/issues/12541) /
  [#12580](https://github.com/anthony-chaudhary/fak/issues/12580) artifact/admission
  work establish preparation, not a native pretrained engine. Source at this
  baseline still has the explicit `dflash2-algorithmic-candidate-rejected-no-trained-layer`
  selector in `internal/model/adaptive_draft_depth.go`.
- **Missing native integration:** exact trained tensor loader, five-layer forward,
  feature/state cache and transactional rollback, target verifier and stop behavior,
  and reachable serving activation. Native execution is owned by
  [#12727](https://github.com/anthony-chaudhary/fak/issues/12727); optimized hybrid
  rollback/panel work coordinates with
  [#13548](https://github.com/anthony-chaudhary/fak/issues/13548). CPU feature-capture
  or loader candidates alone do not close it.
- **Tested / hardware / default:** no pinned full-DFlash2 native serving/quality/
  throughput receipt in this snapshot. Accepted-per-round, draft cost, verify cost,
  quality, and inclusive throughput are unmeasured here. Keep activation unqualified;
  first join the real header and exact target pins to native execution witnesses.

#### DeepSeek-V4.1-Flash + bundled DSpark

- **Bundled trained stages:**
  [`deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277)
  (MIT) carries DSpark `mtp.0`, `mtp.1`, and `mtp.2` stages. The pinned FAK config
  retains `num_nextn_predict_layers=3`, block size 5, target layers `[37, 38, 39]`,
  Markov rank 256, noise token ID 128799, 128 draft routed experts, and top-3 routing.
  These are trained DSpark stages, not interchangeable generic MTP heads.
- **Compatibility/pins:** preserve the draft's original MXFP4 expert and FP8 dense
  format independently of a converted/NVFP4 target. Individual draft shard digests,
  exact header dtypes and offsets, tokenizer/template identities, and load provenance
  remain missing from this inventory. Do not clone the target inventory's norm/head
  dtypes and call them checkpoint truth. A pinned architecture reference is
  [`vLLM dspark.py@763e3f7c911308fa609fb9437642929efe955798`](https://github.com/vllm-project/vllm/blob/763e3f7c911308fa609fb9437642929efe955798/vllm/models/deepseek_v41/nvidia/dspark.py);
  it supplies semantics, not a FAK execution route. The Qwen DFlash2 sidecar is not
  a compatible substitute merely because a dimension matches.
- **Current omission:** `DeepSeekV41Inventory` in `internal/model/v41_inventory.go`
  enumerates embeddings, `layers.0..NumLayers-1`, norm, and head; it omits every
  `mtp.*` tensor. `openV41PackedShardWith` in `internal/model/v41_weights.go`
  ignores names whose inventory lookup fails. Thus a readable bundled checkpoint
  is not evidence that FAK loaded its draft stages.
- **Explicit refusal:** `RefuseDeepSeekV41DSpark` in
  `internal/model/v41_dspark.go` returns `ErrV41DSparkUnsupported`. The separate
  `V41SpecDecodeConfig.Validate` static layout/length gate explicitly does not prove
  losslessness and does not supersede that refusal. Preserve the refusal until
  feature taps, three-stage forward, draft state/cache, anchored verification,
  rollback/stop, and the serving route are qualified.
- **Next slice / qualification:** under the existing V4.1 parent
  [#12640](https://github.com/anthony-chaudhary/fak/issues/12640), enumerate and
  header-validate `mtp.0/1/2` with an explicit draft opener while keeping target
  loading unchanged. Credit such a slice as inventory/loader only. Full native
  DSpark execution, default activation, accepted-per-round, draft/verify cost,
  quality, and end-to-end hardware throughput remain separate required witnesses.

### Support ladder and closure

| Rung | Compiler state | What closes it |
|---|---|---|
| `release-pinned` | complete | all source and artifact pins are present and well formed |
| `descriptor-validated` | complete | `modeldescriptor.Validate` accepts the normalized descriptor |
| `semantic-reference` | pending | every declared semantic and oracle obligation has captured evidence |
| `fak-native` | pending | backend, conformance/refusal-test, and support-doc obligations are witnessed |
| `optimized` | pending | quality-constrained, end-to-end performance obligations are witnessed |

`registration_closure.open` lists every unresolved obligation as `kind:id`; it remains open in the
intake packet by design. The packet is an immutable work contract, not a support-promotion receipt.

### Typed refusal before allocation

Semantic deltas use a closed axis/value vocabulary. An unknown axis or value returns
`UNKNOWN_SEMANTIC_DELTA`; two values for one axis return `CONTRADICTORY_SEMANTIC_DELTA`. Both carry
the unresolved `axis` in a `fak.new-model-refusal/1` JSON object, exit nonzero, and emit no scaffold
or executable behavior. Invalid pins, malformed descriptors, and missing obligation categories also
refuse before allocation. Repair the pinned manifest; do not translate a refusal into a generic
external runtime.

After the packet is accepted, answer the architecture question.

## The one decision

> **Is the new model a variant of an architecture family fak's in-kernel engine already runs, or a
> genuinely new topology?**

| | Fast path — **alias onto an existing family** | Scaffold path — **new topology** |
|---|---|---|
| When | Same backbone (attention + FFN + norm placement) as a family fak runs; differs only in scale/hparams | New norm placement, new attention or FFN mechanism, no existing family fits |
| Examples in-tree | Kimi K2/K3 → `deepseek2`; Bonsai / qwen3.6 → `qwen35`; DeepSeek-V2/V3 spellings → `deepseek2` | Gemma sandwich-norm, OLMo2 post-norm, gpt-oss MoE — the families already scaffolded |
| Work | One `case` in `canonicalGGUFArch` + a recognition test | `fak new-model` → resolver spec, materializer, forward, conformance row |
| New forward code | **None** — reuses the family's forward | Yes — implement and prove the block |
| Cost | Minutes | Hours to days, plus a real-checkpoint oracle |

Most frontier releases are the **fast path**. A "new" flagship is almost always a scaled or lightly
modified member of an existing family (Kimi K2 is a wider DeepSeek-V3; a new Qwen MoE is a Qwen
variant). Reach for the scaffold only when no family's block matches.

## Orient: the two recognition seams

fak recognizes a model at two distinct layers. Know which one you are touching before you edit.

1. **GGUF arch-string seam** — `internal/ggufload/gguf_config.go`. `canonicalGGUFArch` (`:287`)
   normalizes the file's `general.architecture` string; `archUsesMLAMoELayout(...)` (`:237`) gates the
   config applier `applyGLMMoeDsaConfig` (`:500`). This is where a GGUF-distributed model (the
   llama.cpp ecosystem — Kimi, GLM, DeepSeek, most open-weights releases) is classified. `File.Config()`
   runs `canonicalGGUFArch` on the way to a `model.Config`.
2. **Config + forward seam** — `internal/model`. `Config.archFamilyKey()` + the `is<Family>()`
   helpers (`config.go`), the tensor-name resolver `resolveSpecFor` (`tensor_resolver.go`), and
   `ClassifyForwardPath` (`arch_support.go:121`) which returns the `ForwardPathKind`
   (`ForwardGLMDsaMLA`, `ForwardQwen35GDN`, …) that the forward dispatch and the serving preflight key
   on.

The fast path lives almost entirely in seam 1. The scaffold path edits seam 2.

The live serving boundary exercises both: `preflightServeBackendForwardWith`
(`cmd/fak/serve_backend_preflight.go`) opens only the GGUF header, derives the config through
`File.Config()` (so `canonicalGGUFArch` runs), then **refuses** an unsupported model via
`ValidateBackendForwardConfig` before a single token is decoded. Recognition that fails here is an
honest refusal, not a silent mis-route — which is exactly what your recognition test protects.

## Fast path — alias onto an existing family

Worked precedents, each a single small commit with a recognition test and no forward changes:

- **Kimi K2/K3 → `deepseek2`** — `internal/ggufload/gguf_kimi_arch_test.go`. Kimi is a scaled-up
  DeepSeek-V3 (MLA latent attention + DeepSeekMoE, 384 routed experts vs 256). Its GGUFs already
  declare `deepseek2`; `canonicalGGUFArch` additionally normalizes the Moonshot-branded spellings.
- **Bonsai / qwen3.6 → `qwen35`** — `internal/ggufload/gguf_bonsai_arch_test.go`. Same hybrid
  Gated-DeltaNet backbone as qwen35.

### Steps

1. **Confirm the family.** Read the model's `config.json` / HF architecture. Compare its attention
   (MLA? GQA? linear/recurrent?), FFN (dense? MoE — which router: softmax-top-k, sigmoid+bias?), and
   norm placement against the candidate family's forward in `internal/model`. Write down *why* they are
   the same family — the alias is a claim that they are.
2. **Add the spelling(s)** to `canonicalGGUFArch` — one `case` mapping every documented and plausible
   spelling onto the family's canonical arch string. Keep the metadata-key **prefix** the file's own
   spelling; only the arch string is normalized. In the comment, state the family equivalence, any
   axis that is scaled or new, and — for an unreleased model (e.g. a future K3) — a re-pin caveat that
   the tensor/metadata spellings are forward-looking until validated against a real header.
3. **Write the recognition test** (copy the Kimi or Bonsai file). It is a triplet:
   1. **arch normalization** — every spelling maps to the canonical arch; an unrelated arch is
      untouched (no over-normalization outside the brand's namespace);
   2. **config derivation** — a synthetic header keyed on the model's **own** metadata prefix derives
      a `Config` whose family axes (MoE counts, MLA ranks, …) are populated from the raw-prefixed keys;
   3. **classification** — `model.ClassifyForwardPath(cfg, nil)` returns the family's `ForwardPathKind`
      (not a refusal).
4. **Kill any cap myths.** If the family had a known upstream blocker (llama.cpp's 256-expert cap was
   the real Kimi K2 blocker), assert in the test that fak has no such cap — `expert_count` reads
   generically into `cfg.NumExperts`.

### What the fast path does **not** touch

No new `resolverSpec`, no new forward, no new materializer, and **no new conformance row** — the
target family's row already covers the shared forward. Adding one would duplicate the family, not the
model.

## Scaffold path — a genuinely new topology

```
fak new-model --family <name> --topology <prenorm|postnorm|parallel|identity> [--dry-run] [--json]
```

The scaffold (`internal/newmodel`) prints the exact edits and generated code for the four seam-2
touch points:

1. `internal/model/config.go` — the `is<Family>()` helper (keyed on `archFamilyKey()`).
2. `internal/model/tensor_resolver.go` — the `resolverSpec` (tensor-name map) + the `resolveSpecFor`
   case.
3. `internal/model/weights.go` — the materializer wire in `newModel`.
4. `internal/model/family_conformance_test.go` — a `familyConformanceTable` **row** (see below), added
   `supported:false` so the family shows as an unimplemented (skipped) row until you wire the forward.

Then implement the family-specific block, flip the conformance row to `supported:true` once the
synthetic forward runs without panic, and add the numeric oracle when a checkpoint exists.

## Validation ladder (both paths)

Run in order. Each rung proves more and costs more; stop where the risk for your change is retired.

1. **Package build + recognition/unit test** — `go test ./internal/ggufload/ ./internal/model/`. For
   the fast path, the recognition triplet passing here is the bulk of the proof: the model is
   classified and its config is structurally correct.
2. **Weight-free family conformance** — `TestWeightFreeFamilyConformance` in
   `internal/model/family_conformance_test.go` (issue #1081). A table over every registered family
   that runs a synthetic prefill + decode with **no real weights**, in CI, asserting config→topology→
   forward is panic-free and shape-correct. A new topology adds a **row** here (that is what the
   scaffold generates); the fast path inherits the family's existing row.
3. **Forward-path classification / serving preflight** — `ClassifyForwardPath` (`arch_support.go`) is
   the model-layer predicate your recognition test asserts; `preflightServeBackendForwardWith`
   (`serve_backend_preflight.go`) is the live serve boundary that derives the config and refuses an
   unsupported model via `ValidateBackendForwardConfig` before decode. A recognized model must
   classify; an unsupported one must refuse honestly, never silently mis-route.
4. **Coverage & observability** — `fak coverage-matrix` (the generated model × backend grid),
   `fak support` (per-cell rung/regime/next-action read-out), and `fak conformance` (the ABI +
   dogfood verdict matrix) fold the new cell into the tracked surface so growth stays legible.
5. **Real-checkpoint HF numeric oracle** — issue #474, the **permanent gate**. Until a re-exported
   argmax oracle exists on disk, non-Llama numeric correctness is *asserted, not proven*: the
   `TestOptional*Oracle*` tests `t.Skip` when a checkpoint is absent. Mark such a model
   `needs-runtime-witness`; do not claim bit-exactness without the oracle.

## Honest boundaries

- **Recognition ≠ numeric parity.** The fast path proves a model is classified and its config is
  structurally sound. It does **not** prove the forward is bit-exact. A Kimi/DeepSeek MLA model
  classifies and derives a correct config, but its `family_conformance` row is `supported:false` until
  a real checkpoint drives the MLA tensors beyond the synthetic fixture — the S8 boundary (#25), not a
  regression. See [model-arch seam status (#487)](notes/model-arch-seam-status-487.md).
- **Accelerated hot paths lag the proof path.** HAL/Metal/quant-batch decode and `batch.go` still
  `requirePreNorm`-panic on non-PreNorm topologies (#487 S4). A new non-PreNorm family runs on the
  scalar proof path only until those copies are generalized.
- **Do not over-normalize `canonicalGGUFArch`.** An alias asserts two archs are the *same family*. A
  wrong alias silently mis-routes a model onto an incompatible forward. Normalize only within a brand's
  real namespace, and prove the non-over-normalization case in the test.

## Related

- [Zero-day model-onboarding study](notes/CONCEPT-STUDY-ZERO-DAY-MODEL-ONBOARDING-2026-08-27.md)
  — pinned vLLM/SGLang evidence and the release-to-descriptor design decision behind this compiler.
- [Models supported by fak](supported/models.md) — the two meanings of "supported"; the in-kernel grid.
- [Model-arch seam status (#487)](notes/model-arch-seam-status-487.md) — the stage-by-stage,
  file:line status of the seam this playbook rides.
- [Combinatorial-growth epic (#1079)](notes/COMBINATORIAL-GROWTH-EPIC-2026-06-27.md) — the epic that
  shipped `fak new-model`, the weight-free conformance contract, and the generated coverage matrix.
