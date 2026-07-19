---
title: "Adding a new model to fak — the fast path and the scaffold path"
description: "A maintainer playbook for recognizing and validating a new model architecture in fak's in-kernel engine. One decision splits the work: a new frontier model that is a variant of a family fak already runs is a one-line arch alias plus a recognition test (minutes); a genuinely new topology is a `fak new-model` scaffold plus a forward implementation and an oracle (longer). Grounded in the Kimi and Bonsai aliases now in the tree."
---

# Adding a new model to fak

This page is the procedure for making fak's **in-kernel reference engine** recognize and run a
new model architecture. It does **not** cover fronting a model through the gateway — any model your
upstream serves is already fronted unchanged (see [supported models](supported/models.md), Layer 1).
This is Layer 2: the pure-Go forward pass.

The whole playbook turns on one question, so answer it first.

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

- [Models supported by fak](supported/models.md) — the two meanings of "supported"; the in-kernel grid.
- [Model-arch seam status (#487)](notes/model-arch-seam-status-487.md) — the stage-by-stage,
  file:line status of the seam this playbook rides.
- [Combinatorial-growth epic (#1079)](notes/COMBINATORIAL-GROWTH-EPIC-2026-06-27.md) — the epic that
  shipped `fak new-model`, the weight-free conformance contract, and the generated coverage matrix.
