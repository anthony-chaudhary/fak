---
title: "SGLang upstream study — 2026-08-27"
description: "Study of SGLang upstream source tree and forge capture receipt."
---

# SGLang upstream study — 2026-08-27

**Verdict:** the pinned tree is complete, every declared source class and all 12 candidates are accounted for, and no unduplicated implementation ticket was created. The forge corpus is **partial, not complete**: both capture attempts paginated the six endpoint classes, but `fak study-forge` rejected cross-endpoint PR identity consistency.

## Pin and capture receipt

- Repository: `sgl-project/sglang`
- Revision: `536f570e6692eec0656ef9689db7591ca1d0e0a7`
- Inclusive cutoff: `2026-08-27T10:00:00Z`
- Recursive tree: 9746 entries, `truncated=false`
- Forge records captured: 36758 (issues=7163, pulls=29053, discussions=399, releases=59, labels=83, milestones=1)
- Raw payload: 165249827 bytes, `sha256:bf45531c67368675a2919013aafda770da21a053852f8f1eb2a3632c138f61c4`; retained in allocated scratch, not committed.
- Validation: `fak study-forge validate` refused status `partial`. Exact cause: `only_in_mixed=29033`, `only_in_dedicated=29053`, `total=58086`, policy limit `1000`. The differing endpoint numeric IDs prevent a responsible completeness claim even though PR numbers/node IDs correspond.

The complete receipt is gzip-compressed for transport (33.8 MB; 165.2 MB expanded): [`../inventory/sgl-project-sglang-corpus.jsonl.gz`](../inventory/sgl-project-sglang-corpus.jsonl.gz). Full class accounting is in [`source-completeness.json`](source-completeness.json).

## Candidate ledger

| Candidate | Disposition | Existing fak work | Representative upstream evidence |
|---|---|---|---|
| `scheduler-admission` | ADAPT | #35, #36, #8395 | https://github.com/sgl-project/sglang/pull/17026 |
| `radix-prefix-reuse` | COPY | #3889, #3890, #3891, #41, #8395 | https://github.com/sgl-project/sglang/issues/26618 |
| `chunked-prefill-continuous-batching` | ADAPT | #36, #282, #8395 | https://github.com/sgl-project/sglang/pull/35300 |
| `prefill-decode-disaggregation` | INTEGRATE | #28, #29, #37, #53, #79 | https://github.com/sgl-project/sglang/pull/35224 |
| `speculative-decoding` | WATCH | #23 | https://github.com/sgl-project/sglang/pull/21272 |
| `structured-generation` | INTEGRATE | #26, #8382 | https://github.com/sgl-project/sglang/pull/28804 |
| `multimodal-moe` | WATCH | #25, #290, #399, #4875, #5777 | https://github.com/sgl-project/sglang/pull/27602 |
| `distributed-execution` | ADAPT | #25 | https://github.com/sgl-project/sglang/issues/22084 |
| `observability` | INTEGRATE | #216, #5629, #5631, #6801 | https://github.com/sgl-project/sglang/pull/23169 |
| `kernel-runtime-integration` | REJECT | #8395 | https://github.com/sgl-project/sglang/issues/26715 |
| `failure-reliability` | ADAPT | #8382, #8395 | https://github.com/sgl-project/sglang/pull/32118 |
| `compatibility-benchmark` | INTEGRATE | #39, #44, #6473, #6474 | https://github.com/sgl-project/sglang/tree/536f570e6692eec0656ef9689db7591ca1d0e0a7 |

Dispositions mean: **COPY** a narrow proven mechanism while retaining fak ownership; **ADAPT** its invariant to a fak seam; **INTEGRATE** only through an explicit governed adapter; **WATCH** until the operating envelope justifies work; **REJECT** wholesale adoption or silent fallback. SGLang never becomes an automatic recovery path for native inference.

The machine ledger is [`candidate-ledger.json`](candidate-ledger.json). It records operating envelope, next-best alternative, evidence, and ticket mapping for every candidate.

## Dedup result

[`dedup-map.json`](dedup-map.json) accounts for all 271 issues returned by the repository-wide `sglang` tracker query and explicitly maps every candidate to existing work. Direct anchors include #23–#44, #216/#282/#290/#322, #3889–#3891, #5629/#5631, #6473/#6474, #6801, #8382, and #8395.

**New issues created: none.** The study found no genuine unduplicated gap that cleared the existing serving, compatibility, parser, cache, distributed, multimodal, observability, or benchmark tickets.

## Completeness boundary

- Complete: pinned recursive tree; discussions; releases; labels; milestones.
- Endpoint-complete but corpus-contract partial: issues and pulls.
- Derived and accounted: roadmap, benchmarks, failure/reliability history.
- Not claimed: semantic perfection of lexical classification, durable retention of allocated raw scratch, or a valid complete `fak-studyforge-corpus/1` receipt.

## Deterministic validation

Run:

```powershell
gzip -dc docs/research/inventory/sgl-project-sglang-corpus.jsonl.gz > $TMPDIR/sglang-corpus.json`npython -m json.tool $TMPDIR/sglang-corpus.json > /dev/null
Get-ChildItem docs/research/sglang-2026-08-27/*.json | ForEach-Object { python -m json.tool $_.FullName > $null }
fak study-forge validate --receipt $TMPDIR/sglang-corpus.json  # -> valid
```

See [`validation-report.json`](validation-report.json) for the executed checks and checksums.
