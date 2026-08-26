---
title: "Issue #8624 — observed small-model Ultracode cache frontier"
description: "Verdict: on qwen2.5:0.5b through the sanctioned fak-realmodel node, every bounded agentic cell at widths 1, 2, 4,"
---
# Issue #8624 — observed small-model Ultracode cache frontier

**Verdict:** on `qwen2.5:0.5b` through the sanctioned `fak-realmodel` node, every bounded agentic cell at widths 1, 2, 4, and 8 retained the frozen accepted outcome while reducing scoped context and reusing a shared prefix. The evaluator therefore hill-climbed both multi-agent modes through width 8.

## Replay

```powershell
fak ultracode bench --scenario access-frontier `
  --scenario-input docs/_witnesses/issue-8624-ultracode-smallmodel/access-frontier.json `
  --widths 1,2,4,8 --json
```

Files:

- `source-artifact.json` — source-linked observed campaign: node/runtime/model, frozen task digest, bounded scout/writer/reviewer roles, accepted responses, Ollama `cached n_tokens` excerpts, and request usage telemetry. Its `raw_capture_sha256` binds the compact checked-in receipt to the complete raw capture retained during the run.
- `access-frontier.json` — evaluator input (`evidence_kind: observed_run`).
- `report.json` — replayed `fak-ultracode-access-frontier/1` result.

## Result

| Mode | Width | Full-context counterfactual | Scoped child context | Shared-prefix reads | Total avoided | Verdict |
|---|---:|---:|---:|---:|---:|---|
| scout_writer | 1 | 4,095 | 1,639 | 1,024 | 3,480 | GAIN |
| scout_writer | 2 | 8,190 | 3,280 | 2,628 | 7,538 | GAIN |
| scout_writer | 4 | 16,380 | 6,562 | 5,908 | 15,726 | GAIN |
| scout_writer | 8 | 32,760 | 13,126 | 12,468 | 32,102 | GAIN |
| multi_writer | 1 | 4,095 | 1,641 | 1,024 | 3,478 | GAIN |
| multi_writer | 2 | 8,190 | 3,280 | 2,628 | 7,538 | GAIN |
| multi_writer | 4 | 16,380 | 6,560 | 5,836 | 15,656 | GAIN |
| multi_writer | 8 | 32,760 | 13,120 | 12,252 | 31,892 | GAIN |

The single-agent width-1 control also gained, but it is not a throughput baseline. It establishes that the same harness semantics benefit from scoping and prefix reuse before concurrency is increased.


## Win record: what belongs to fak, and what does not

This is a **combined-system win**, not a claim that fak invented prefix caching.

Across the eight multi-agent cells (`scout_writer` and `multi_writer`, widths 1/2/4/8), the evaluator credited **117,410 avoided input-token accesses**:

| Source | Avoided token accesses | Share of credited total | Attribution |
|---|---:|---:|---|
| Role-scoped micro-contexts | 73,642 | 62.7% | fak's agentic decomposition and context-selection policy |
| Runtime shared-prefix reads | 43,768 | 37.3% | Ollama/llama.cpp's ordinary prompt/KV prefix cache |

The runtime rows (`cached n_tokens`, including “prompt is already in the cache”) prove that prefix reuse happened. They do **not** prove that fak supplied a new radix-tree implementation. The portable fak contribution demonstrated here is that bounded scout/writer/reviewer contexts omit irrelevant broad history while preserving one stable project-contract prefix that an existing runtime cache can reuse.

The current artifact supports three claims:

1. **Scoping works independently in the accounting:** `full_context_input_tokens - scoped_context_input_tokens` is positive in every accepted multi-agent cell.
2. **Ordinary prefix reuse adds independently in the accounting:** runtime-authored `cached n_tokens` is positive in every accepted multi-agent cell.
3. **The composition scales in this envelope:** accepted outcomes remain equal through width 8 while both terms grow.

It does **not yet identify a causal fusion bonus** beyond `scope_avoided + prefix_read`: there is no 2×2 campaign with scoping on/off and prefix caching on/off. Nor does it test fak's Random Access Cache, addressed non-prefix reuse, cross-request planning, repaired spans, or a fak-owned radix index. Those are separate innovations and require separate witnesses rather than inheriting this result.

### Required next decomposition

Use the same frozen task and outcome digest in a factorial campaign:

| Cell | Context policy | Runtime prefix cache | What it identifies |
|---|---|---|---|
| A | full context | off/cold | no-reuse control |
| B | full context | on/warm | generic radix/prefix contribution |
| C | scoped micro-context | off/cold | fak scoping contribution |
| D | scoped micro-context | on/warm | combined result |

Report `B-A`, `C-A`, and `D-A`; call any fusion term only as `D - B - C + A`, with confidence intervals and equal-outcome gating. Add RAC/addressed-reuse cells only after this ordinary-prefix baseline is green.

## Measurement boundary

- `full_context_input_tokens` is an observed full-context prompt count multiplied by equal agent width; it is a context-access counterfactual, not a second latency run.
- `scoped_context_input_tokens` is Ollama's per-request `prompt_eval_count` summed across bounded role contexts.
- `shared_prefix_read_tokens` is the largest Ollama llama-server `cached n_tokens` value for each task in the source log. This is runtime-authored KV/prompt-cache telemetry, not a value inferred from wall time.
- All responses normalized to the same `ACCEPTED` outcome digest before savings were credited.
- The claim excludes raw single-request throughput, traditional batching, billed tokens, and spend. It applies only to this observed model/runtime/task envelope.

## Scoped-prefix regression contract (#8680)

The 62.7% scoped-context / 37.3% runtime-prefix attribution above is bound to
`internal/ultracodebench/testdata/scoped-prefix-regression-v1.json` at
**corpus-row: observed-positive-qwen25-05b**. The affected-package test loads
that row and the five predeclared negative/abstain controls without a live model
call; removing the row or this reference fails the fast gate.

Refresh the observed row quarterly, or sooner when the evaluator, harness,
runtime, model, tokenizer, task, or cache posture changes. Promotion evidence is
a second versioned campaign inside the bounded ranges with equal accepted
outcomes and disjoint authoritative accounting. Demote or retire the claim when
a refresh abstains, leaves the ranges, or loses replayable telemetry. The
invalidating assumption is that scoped omission and runtime prefix reads remain
separable under the named warm-prefix factorial cache posture; a cache reset or
overlapping token attribution invalidates the comparison.
