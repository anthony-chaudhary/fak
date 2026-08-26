# Architecture quality per active byte

## Decision record

- **For:** architecture researchers choosing which memory-efficiency hypothesis deserves a measured migration experiment.
- **Problem:** parameter count alone hides recurrent state and envelope-dependent KV memory, while unlike evaluations invite false rankings.
- **Today:** GQA, MQA, MLA, SSM, conditional-depth, and sparse-attention ideas arrive from different sources with estimates that are not directly comparable.
- **Better because:** one conservative denominator and an exact comparability key make unsupported comparisons visibly unranked instead of silently normalizing them.
- **Witness:** `go test ./internal/archrank` loads `internal/archrank/testdata/candidates.json`, validates every row, reproduces a two-row synthetic control ranking, and proves estimates and comparison-key mismatches remain unranked.

This is an **Enabling** benchmark method, not evidence that any named architecture is better. The synthetic controls are method witnesses only: their invented byte counts and quality values are **not model measurements or model claims**. The literature rows are hypotheses for future migration work. They remain estimated and unranked even when a URL describes the underlying technique.

## Record and formula

Each candidate records:

- `active_weight_bytes`: bytes of weights read for the evaluated inference path.
- `state_bytes`: bytes of persistent non-KV runtime state needed at the envelope.
- `kv_bytes_at_envelope`: bytes of KV cache at the declared envelope.
- `quality`: the finite, non-negative, dimensionless value emitted by the named quality metric. This method defines larger values as better; lower-is-better metrics require a separately named, predeclared transformation before they can be supplied.
- `envelope_id`: an opaque identifier for the complete operating envelope. Producers must change it when context length, batch, precision, hardware-relevant memory policy, or another comparability condition changes.
- `quality_metric`: the exact metric identity, including any version or evaluation variant needed to prevent accidental equivalence.
- `quality_source_kind`: how quality was obtained. Only accepted, measured rows are eligible.

All three memory inputs and `active_bytes` use **bytes** (`B`, not bits, parameters, MiB, or GiB):

```text
active_bytes = active_weight_bytes + state_bytes + kv_bytes_at_envelope
quality_per_active_byte = quality / active_bytes
```

The implementation uses non-negative integer byte fields, requires `active_bytes > 0`, and rejects overflow. The score therefore has units of **quality-metric units per byte**. It is useful only inside a comparison group; it is not a universal model score.

## Eligibility and grouping

A row is ranked only when all of the following hold:

1. Its provenance has a non-empty kind and the evidence locator required for that kind: accepted measurements use an artifact locator, while literature hypotheses use an absolute HTTP(S) source URL.
2. Its declared formula is the exact formula above and its fields validate.
3. Its evidence status is accepted and measured rather than estimated.
4. At least one other eligible accepted row has an **exact** match on the tuple `(envelope_id, quality_metric, quality_source_kind)`.

Eligible rows are sorted by `quality_per_active_byte` descending, with a deterministic identifier tie-break. Estimates, formula or provenance failures, unsupported metric directions, singletons, and rows whose envelope, metric, or quality-source kind does not match a comparable cohort are explicitly unranked with reasons. The loader never converts a literature estimate into measured evidence and never merges near-matching group keys.

## Fixture and reproduction

`internal/archrank/testdata/candidates.json` has two intentionally synthetic measured controls sharing one exact comparison key. They provide a stable arithmetic and ordering witness. Additional GQA, MQA, MLA, SSM, conditional-depth, and sparse-attention rows carry literature URLs and migration classes, but are marked estimated so they cannot enter a ranking.

From the repository root:

```bash
go test ./internal/archrank
```

The test fails if loading, provenance validation, formula validation, active-byte accounting, exact-key grouping, deterministic ranking, or explicit unranked reasons regress.

## Interpreting a decision

A higher in-group value means only that the accepted measurement delivered more of that exact higher-is-better quality metric per active byte at that exact envelope. It does **not** establish lower latency, lower total memory, better training efficiency, equivalent output distribution, or superiority outside the group.

Use the result to select the next controlled migration measurement, not to declare an architecture winner. Promote a literature hypothesis only after reproducing it with the same envelope, metric, source kind, byte accounting, and acceptance process as its control. If those conditions cannot be made exact, preserve the result as unranked evidence rather than forcing a comparison.
