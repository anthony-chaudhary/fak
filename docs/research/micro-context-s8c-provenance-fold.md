---
title: "Micro-context S8c provenance-preserving hierarchical fold"
description: "Fixture-backed bounded typed reducers with content-addressed provenance and precise invalidation."
status: observed
last_reviewed: 2026-08-09
---

# S8c: provenance-preserving hierarchical fold

## Verdict

**Observed in the fixture:** the 1,000-record operator can fold typed facts without sending the
original corpus to any reducer and without turning evidence into uncited prose. The reduction tree
uses fan-in eight, content-addresses leaves and intermediate nodes, carries exact coverage/status
counts, deterministic sets and stable top-k, and retains bounded semantic-cluster exemplars plus
named outliers.

Captured artifact:
[`s8c-local-provenance-fold-1000-pass-2026-08-09.json`](../../experiments/microcontext/s8c-local-provenance-fold-1000-pass-2026-08-09.json).

This is a fixture-backed reducer witness. It does not claim arbitrary model summaries are
associative, lossless, or semantically invariant.

## Reproduce

```bash
go run ./cmd/microcontextdemo \
  -provenance-fold-selfcheck \
  -provenance-fold-output /tmp/provenance-fold.json

go run ./cmd/microcontextdemo \
  -verify-provenance-fold /tmp/provenance-fold.json
```

## Witness

- 1,000 leaves reduce through four levels and 144 intermediate nodes; maximum input is eight.
- Every node output has at most 13 citations, below the verifier cap of 64.
- Exact count/set/stable-top-k and bounded cluster results have the same result hash after input
  reversal and under alternate fan-in 13.
- The single security-dissent source and the single abstention source survive as named outliers;
  uncertainty remains an explicit `abstain` count rather than being smoothed into a majority.
- Every one of the ten final citations resolves through an independent source registry.
- Mutating one source recomputes exactly five nodes—the leaf plus its four ancestors—and reuses
  1,139 unaffected content-addressed nodes.
- A deterministic sample of 25 excluded records passes a negative audit.

## Reducer contracts

| Reducer | Safe claim | Required condition |
|---|---|---|
| Count | associative and exact | typed disjoint source coverage |
| Set union | associative and exact | canonical values and deterministic order |
| Top-k | shape invariant | stable score, stable source-ID tie rule, candidates retained to k at every node |
| Semantic cluster | explicitly lossy | exact cluster counts plus bounded exemplars and named outliers |

The semantic reducer is not advertised as lossless. Its contract is narrower: cluster counts are
exact for the fixture taxonomy, exemplars are deterministic, and outliers named by the map stage
cannot disappear. A production model-assisted clusterer would additionally need drift measurement
against an independently labeled sample.

## Steelman perspectives

### Why hierarchy is essential

Flat map/reduce only relocates context overflow into one giant reducer. A bounded tree keeps every
prompt or deterministic operation within a declared input/output envelope, permits subtree cache
reuse, and makes invalidation proportional to tree depth instead of corpus size.

### Why hierarchy is dangerous

Loss compounds. Early semantic summaries can erase a minority observation before the final reducer
can know it existed; tree shape and ordering can alter model prose; and top-k pruning is unsound when
scores are unstable or unseen candidates lack an upper bound. Content hashes prove which bytes were
combined, not that the combination preserved meaning.

### Strong conventional alternative

For counts, sets, grouping, joins, and top-k over stable scores, databases and streaming dataflow
engines are the right reducer. The LLM should receive their typed result, not reimplement them. Model
micro-windows add value only for residual semantic classification, exemplar selection, or ambiguity,
and even there the deterministic envelope should own coverage and provenance.

### Auditability versus privacy

Source-level citations make claims checkable but can retain sensitive identifiers longer than a
plain summary. Production designs may need access-controlled provenance handles, retention limits,
and redacted read-back—not citation removal disguised as privacy.

## Boundary

The witness establishes typed fold mechanics and precise invalidation for #6032. It does not establish
live-model semantic quality or net economics; #6033 must compare the full selector/tool/fold fabric
against tuned server-side filtering, retrieval, and batching baselines.
