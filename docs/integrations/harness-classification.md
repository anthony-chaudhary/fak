---
title: "Contextual harness domain classification"
description: "fak harness classify chooses among the initial closed domains legal, coding, and integrated without a model call or a recurring profile picker."
---
# Contextual harness domain classification

`fak harness classify` chooses among the initial closed domains `legal`, `coding`, and `integrated` without a model call or a recurring profile picker.

```text
fak harness classify --path matter.docx --task "draft deposition brief"
fak harness select --manifest harness.json --path matter.docx --task "draft deposition brief"
```

## Precedence and quiet-launch rule

Classification is explicit-first:

1. `--task-domain`;
2. `--project-domain`;
3. an unexpired, exactly scoped `--choice-file`;
4. deterministic inference from path extension, task tokens, and named signals.

Explicit declarations and valid remembered choices have confidence `1`. Inference auto-selects only when the leading domain has score at least four, a margin of at least two, and at least two independent matching signals. Otherwise the command emits one `decision_request` with reason, choices, and a context key and exits `3`. Nothing launches from an ambiguous domain.

The current classifier is intentionally transparent rather than model-based. Its JSON includes candidate scores and every evidence item/weight. This keeps classification cost bounded, reproducible, and inspectable before tools or adapters load.

## Reversible remembered choices

An operator can resolve one ambiguity and create a short-lived choice:

```text
fak harness classify --path brief.go --task "implement contract brief" \
  --choose legal --choice-out .fak/legal-choice.json \
  --reason "operator confirmed legal matter" --ttl 1h
```

The choice records domain, a context key derived from normalized path plus exact task, expiry, and operator reason. Reuse with `--choice-file`. A different path/task, expired choice, missing reason, or unknown domain fails before selection. Reversal is deleting the choice file; fak never installs it globally or learns away a company policy floor.

## Witness and limits

The committed corpus covers three clear legal, three coding, three integrated, and three ambiguous cases. It reports **0 false auto-selections across 12 authored cases**; this is a regression corpus, not a population accuracy estimate. On the 2026-08-15 Windows AMD Ryzen 9 9950X witness, deterministic classification measured **1,426 ns/op, 1,048 B/op, 23 allocs/op** over 100,000 iterations. The meaningful cost is effectively below a network/model turn, but no productivity claim follows from this microbenchmark.

`TestHarnessSelectClassifiesLegalWithoutCodingLeak` proves a legal task admits citations while excluding the coding-only shell layer. `TestHarnessClassifyCLIExplicitAmbiguousAndRemembered` proves explicit precedence, exit-3 ambiguity, scoped choice creation, and reuse.

This first vocabulary is not universal ontology. New domains require an explicit contract and corpus. Domain classification does not compose typed assets (#6792/#6904), preview privilege changes (#6902), or measure operator-value against tuned host-native profiles (#6903).
