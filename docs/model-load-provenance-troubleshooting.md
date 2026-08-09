---
title: "Model-load provenance: route a run difference"
description: "Tell whether a degraded model run came from the loader, the quantization, or the forward pass, using the fak-model-load-provenance/1 artifact and its algebra."
---

# Model-load provenance — routing a run difference to loader, quant, or forward

**The troubleshooting guide for [#4746](https://github.com/anthony-chaudhary/fak/issues/4746),
written out of root incident [#4273](https://github.com/anthony-chaudhary/fak/issues/4273).**
Artifact + algebra: [`internal/model/loadprovenance.go`](../internal/model/loadprovenance.go) ·
GGUF producer: [`internal/ggufload/gguf_load_provenance.go`](../internal/ggufload/gguf_load_provenance.go) ·
transform contracts: [`internal/ggufload/gguf_transform_contract.go`](../internal/ggufload/gguf_transform_contract.go)
(schema `fak-model-load-provenance/1`).

## The question this exists to answer

During #4273 a Qwen3.6 load produced degraded output, and every fact the runtime
could report — model id, quant mode, tokens/sec — was **identical between the
broken load and the fixed one**. The defect lived entirely in loader semantics:
GGUF `blk.*.ssm_a` stores the already-transformed negative decay `-exp(A_log)`,
fak's canonical tensor is the pre-transform `A_log`, and the forward applied
`exp` twice. With nothing reporting what the loader *did* to the bytes, the
investigation had to guess between sampler, prefill, decode, quant, and loader —
and spent itself on the wrong four.

The load-provenance artifact is the missing report: a compact, publish-safe
record of the loader's semantic transforms, content-addressed so a claim about a
run can be bound to the loader semantics that actually produced its weights.

**Use this guide when two runs of the same model disagree** — different output
quality, different numerics, a regression across a rebuild — and you need to know
which subsystem to open first.

## Step 1 — get an artifact for each run

```go
p, err := ggufFile.LoadProvenance(ggufload.LoadProvenanceScope{
        ModelDigest: "sha256:…", // over the model file bytes
        ModelBytes:  size,
        LoaderRev:   "fak-loader@<commit>",
        Quant:       "Q4_K",
        ForwardPath: model.ForwardQwen35GDN,
})
```

`LoadProvenance` is **fail-closed**: it validates the assembled artifact and
returns an error rather than a record with a blank loader revision, because
inconclusive evidence that still gets attached to a claim is worse than none — it
looks like provenance while proving nothing.

`p.Digest()` is the content address to record alongside the run. `p.JSON()` is
the human-readable form. Both are safe to attach to a public failure bundle
unedited: no field in the record has a type that could hold a prompt, a
filesystem path, or a raw weight.

## Step 2 — diff them, and read the FIRST line

```go
for _, d := range model.DiffLoadProvenance(runA, runB) {
        fmt.Println(d)
}
```

Each delta prints as one line:

```
transforms.invert-neg-exp-decay/ssm_a: "canonical=linear_attn.A_log tensors=48 layers=48 …" -> "" [investigate loader]
```

`field` is the canonical dotted path, `A`/`B` are the two readings (a side that
lacks the entry reads `""`), and the bracket is the **investigation area** the
difference implicates. Deltas come back in a fixed order — model-bytes, loader,
quant, forward — deliberately: **the first line is the one that invalidates the
rest.** Do not triage a `quant` delta while a `model-bytes` delta is still open;
the runs did not load the same artifact, so nothing downstream is comparable.

An empty diff is itself an answer: the two loads produced identical loader
semantics, quantization, and forward path, so any behavioral difference lies in
the sampler, the prompt, or scheduling — not here.

## Step 3 — the routing table

The areas are a closed set. That is the point: a delta arrives pre-triaged
instead of as JSON to eyeball, and an open vocabulary would let a caller invent a
fifth area no runbook covers. This table is the prose projection of the routing
in `DiffLoadProvenance`, which is executable and therefore cannot drift from the
fields it explains.

| Fields that differ | Area | What it means | Open this first |
|---|---|---|---|
| `model_digest`, `model_bytes`, `gguf_version`, `manifest_digest` | `model-bytes` | The runs did not load the same artifact. | Reconcile the checkpoints. Stop — every downstream delta is uninterpretable until they agree. |
| `gguf_arch`, `loader_rev`, `transforms.*`, `domain_checks.*`, `tensor_summaries.*` | `loader` | Same bytes, different **semantics**. This is the #4273 class. | The arch's transform contract in `internal/ggufload/gguf_transform_contract.go`, and the canonicalizer in `gguf_tensor_canonical.go`. |
| `quant` | `quant` | Loader semantics agree, so the mapping is right and the numerics are suspect. | Dequant kernels and per-tensor quant assignment — **not** the loader. |
| `forward_path` | `forward` | Bytes, loader, and quant agree, but the runtime chose a different token mixer. | `model.ClassifyForwardPath` and the selected mixer — **not** the loader. |

### Reading a `loader` delta

A `transforms.<id>/<external>` reading is
`canonical=… tensors=N layers=N lossless=… invertible=… domain_validated=…`.
Three shapes recur:

- **A transform vanished** (`"…" -> ""`). The loader stopped applying a mapping
  it used to apply. This is exactly #4273: the ssm_a decay inversion disappearing
  means the forward now sees `-exp(A_log)` where it expects `A_log`. Check
  whether the arch spelling still resolves to a contract —
  `TensorTransformContractsForArch` is keyed on the canonicalized
  `general.architecture`, so a new vendor spelling silently yields *no*
  transforms rather than an error.
- **Counts moved** (`tensors=48 layers=48` -> `tensors=47 layers=47`). The
  transform is still declared but did not fire on every tensor it should have.
  Suspect a tensor-name change in the exporter.
- **Same transform ids, a moved `tensor_summaries.*` hash.** The declared
  mapping is unchanged but its *implementation* produces different values — a
  numerics change inside the transform itself. The hash covers the name, the
  shape, and every element's IEEE-754 bit pattern, so it also catches a
  transform that preserves the value *set* but permutes it (which is precisely
  what the value-head deinterleaves do).

`domain_checks.<transform>` reads `tensors=N rejected=N expected=<domain>`. A
non-zero `rejected` is the artifact form of a loader refusal; its `first_failure`
is the publish-safe evidence string (tensor name, transform id, element index,
expected domain — the offending **value** is deliberately withheld, because it is
a raw weight element and belongs in the operator's error, never in a record that
may be published).

### When the loader refuses outright

A source-domain violation does not produce a subtle delta — it fails the load
with a typed `*model.SourceDomainError` naming all four facts:

```
model: tensor model.layers.0.linear_attn.A_log element 1 = 0.5 violates transform
invert-neg-exp-decay source domain (want finite, strictly negative -exp(A_log))
```

A **non-negative** or NaN "decay" is a value only plausible in the *canonical*
(`A_log`) domain — i.e. the tensor was already un-transformed, or the exporter
wrote the wrong domain. That is the #4273 mistake caught at load time instead of
at generation time. `-Inf` is refused too: it reads as negative to a naive
`>= 0` test, but `log(-(-Inf))` is `+Inf`, so it would canonicalize into an
infinite `A_log` rather than being refused at the domain it violates.

## Worked example — the #4273 regression

Two runs of the same Qwen3.6 checkpoint, one good, one degraded:

```
transforms.invert-neg-exp-decay/ssm_a: "canonical=linear_attn.A_log tensors=48 layers=48 lossless=true invertible=true domain_validated=true" -> "" [investigate loader]
```

One line, and the investigation is over before it starts: same bytes, same quant,
same forward path, and the decay inversion that ran on 48 layers in the good run
ran on none in the bad one. Open the ssm_a contract. Do not profile the sampler.

To ask the question directly without diffing:

```go
tensors, layers, ok := p.TransformTensors(ggufload.TransformInvertNegExpDecay)
```

`ok=false` means the transform did not fire at all on this load. A composite
transform matches on any of its `+`-joined components, so asking for
`invert-neg-exp-decay` finds it inside
`value-head-deinterleave+invert-neg-exp-decay`.

## What the artifact deliberately does not contain

- **No timestamp, host, device, file path, or wall-clock cost.** Those are run
  scope, not load scope. Their absence is what makes the digest *deterministic*:
  identical model bytes plus an identical loader revision serialize
  byte-identically on any host, so two operators can compare digests across
  machines. Run-scope facts belong in `internal/provenance.RunManifest`, which
  can carry this digest as one of its recorded facts.
- **No prompts, no raw weights.** Enforced by construction rather than by
  review: there is no `[]float32`, no `[]byte`, and no open map anywhere in the
  record. `SummarizeTransformedTensor` is the only entry point that ever sees
  weight values, and it accepts weights and returns none — a shape, a finite
  flag, a formatted min/max, and a one-way hash.
- **No fabricated range.** `min`/`max` cover the *finite* elements only and are
  both empty when a tensor holds no finite element, rather than reporting a `0`
  bound that is not an element of the tensor.

## Wiring status — what is live today

Stated plainly so nobody reads a capability into this guide that is not there:

- **Live.** `(*ggufload.File).LoadProvenance` builds the artifact from a parsed
  GGUF header — the tensor directory and metadata KV block only. A
  multi-hundred-GB checkpoint answers "did the ssm_a decay inversion fire, and on
  how many layers?" for the cost of a header parse, touching no weight byte.
  `model.DiffLoadProvenance` routes the deltas. The ssm_a domain guard in
  `normalizeCanonicalTensorData` refuses through `model.CheckSourceDomain` on the
  live load path.
- **Empty by design in the header-only producer.** `domain_checks` and
  `tensor_summaries` are *value* evidence and cannot be honestly synthesized from
  a header. Emitting `DomainCheck{Rejected: 0}` there would assert a validation
  that never ran — the exact species of unwitnessed claim this artifact exists to
  kill. They stay empty until a weight-touching pass appends them.
- **Live in the run-evidence schema.** `internal/provenance.RunManifest` *does*
  carry the digest: field `load_provenance`, refused by `Validate` unless it is a
  `sha256:<64 lowercase hex>` content address, and ordered in the fingerprint
  directly after `model` and ahead of every downstream fact. So two runs that
  agree on model, tokenizer, backend, hardware, seed, and decode params but were
  built under different loader semantics are **not** `Equivalent`, and `Compare`
  localizes the divergence to `load_provenance` rather than passing them as the
  same run. That seam is witnessed end to end (see the last two rows below),
  which matters because neither side can check the other: `provenance` is
  stdlib-only and never imports the loader, and `model` cannot see the evidence
  schema at all.
- **Not wired yet.** Two things genuinely remain. (1) **No command renders or
  diffs the artifact** — `fak info` has no provenance mode, so Steps 1 and 2
  above are a Go API, not a CLI an operator can run against two run directories.
  (2) **Nothing in production constructs a `RunManifest`**, so the
  `load_provenance` field is *enforced but unpopulated*: the schema will refuse a
  malformed digest, but it cannot supply a missing one. Until a producer lands,
  the digest has to be recorded by whoever performs the load.

## Witnesses

| Claim | Test | Package |
|---|---|---|
| A Qwen3.6 load reports the ssm_a transform with tensor/layer counts | `TestLoadProvenanceReportsQwen36SSMADecayInversion` | `ggufload` |
| The artifact is deterministic for identical bytes + loader revision | `TestLoadProvenanceDeterministicPerBytesAndLoaderRev` | `ggufload` |
| Recording the same facts in a different order yields the same digest | `TestProvenanceDigestIsOrderIndependentAndChangeSensitive` | `model` |
| An invalid source-domain value fails with all four facts, on the live load path | `TestSSMADomainRefusalNamesAllFourFacts` | `ggufload` |
| A vanished transform routes to `loader`, not `model-bytes` | `TestLoadProvenanceDiffRoutesTransformDeltaToLoader` | `ggufload` |
| Every delta routes to its own investigation area | `TestDiffLoadProvenanceRoutesEachDeltaToItsInvestigation` | `model` |
| The artifact leaks no weights or paths | `TestProvenanceArtifactLeaksNoWeightsOrPaths` | `model` |
| An incomplete artifact is refused rather than published | `TestLoadProvenanceRefusesIncompleteScope` | `ggufload` |
| A real header parse's digest is accepted by the run-evidence content-address check | `TestLoadProvenanceDigestIsAcceptedByRunEvidence` | `ggufload` |
| Run evidence diverges on loader semantics alone, localized to `load_provenance` | `TestRunEvidenceDivergesOnRealLoaderSemantics` | `ggufload` |

```
go test ./internal/model ./internal/ggufload -count=1
```

### What the Qwen3.6 witness does and does not prove

`TestLoadProvenanceReportsQwen36SSMADecayInversion` parses a **synthetic,
header-only GGUF fixture** built in-test (`writeLoadProvenanceGGUF`) whose
`general.architecture` is the vendor spelling `qwen3.6` — not a real Qwen3.6
checkpoint. It is not a mock: the artifact is produced by the shipping
`(*ggufload.File).LoadProvenance` from a genuinely parsed header, and the test
asserts the fixture is payload-free (`TensorDataOffset` ≥ file size) so every
recorded fact is *proven* to come from the header alone.

What it therefore does prove: the producer names the `ssm_a` decay inversion,
canonicalizes `qwen3.6` onto the `qwen35` family, counts distinct tensors and
distinct layers, gives a model-global transformed tensor a layer count of 0,
and omits identity mappings. Because the producer is header-only **by
construction**, none of that needs weights — a real checkpoint would exercise
the identical code path with a larger tensor directory.

What it does not prove: the tensor/layer counts of any *shipped* Qwen3.6
release (the fixture's layer count is chosen by the test), and nothing about
`domain_checks` or `tensor_summaries`, which are value evidence a header-only
producer deliberately leaves empty. Confirming the first against a real
checkpoint needs only that checkpoint's **header**, not its weights.
