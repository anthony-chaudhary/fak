---
title: "Micro-context S8f leakage-controlled GitHub issue corpus"
description: "A frozen 1,000-record non-generated issue corpus, separated answer bundle, deterministic split, independent grader, and blind oracle witness for tuned-baseline work."
status: observed
last_reviewed: 2026-08-09
---

# S8f: non-fixture corpus and independent grader

## Verdict

A real 1,000-issue GitHub snapshot now replaces generated benchmark records for the next
falsification stages. Candidate pipelines receive the authentic source payload: title, body, state,
labels, timestamps, closure time, URL, issue number, and a deterministic `train | tune | test`
assignment. Derived relation targets, cue facts, and aggregate gold outputs live in a separately
hashed answer bundle consumed by the grader. Keeping source metadata visible is essential for a fair
SQL/search baseline; hiding it would turn an exact query into an artificial prediction task.

This is a **corpus/grader witness**, not a model-quality or performance result. It does not establish
that micro-context execution wins; it makes that claim falsifiable on held-out real records.

Artifacts:

- [`s8f-github-issues-public-2026-08-09.json`](../../experiments/microcontext/s8f-github-issues-public-2026-08-09.json) â€” candidate-visible input;
- [`s8f-github-issues-answers-2026-08-09.json`](../../experiments/microcontext/s8f-github-issues-answers-2026-08-09.json) â€” separately hashed grader input;
- [`s8f-github-issues-report-2026-08-09.json`](../../experiments/microcontext/s8f-github-issues-report-2026-08-09.json) â€” provenance, leakage checks, counts, and blind oracle grade.

## Captured corpus

| Property | Witness |
|---|---:|
| Real public issues | 1,000 |
| Train / tune / held-out test | 212 / 108 / 680 |
| Open / closed | 380 / 620 |
| Parsed issue-reference edges | 2,037 |
| Explicit duplicate edges | 12 |
| Contradiction/conflict cue facts | 102 |
| Public-scrub affected records / matches | 8 / 30 |
| Public corpus SHA-256 | `0740c911815eba7e44c63a14e021f418b66e8a42556d40f14b8b4cd58f173827` |
| Answer bundle SHA-256 | `af827ddca1063c602153f80b14523bda8d07e1a7e909a5c487616f2702eda05d` |

Exact duplicate-title/body records and all records connected by in-corpus issue references are
first grouped into components. Each component is assigned by `SHA-256(minimum issue number) mod 10`:
two buckets train, two tune, six held-out test. This prevents direct duplicates and linked issue
neighborhoods from straddling tune/test. Assignment occurs before tuning and is independent of
source ordering.

## Answer contract

The grader covers:

1. held-out state and label facts;
2. issue references and explicit duplicate targets;
3. contradiction/conflict **cue detection** (lexical cues, not adjudicated semantic contradiction);
4. exhaustive state and label counts;
5. newest top-10 and most-recently-updated top-10;
6. exact record citations and corpus-digest binding.

A candidate submission carries the public corpus digest, one typed answer per record, and aggregate
answers. The grader compares only held-out per-record facts, compares exhaustive aggregates, rejects
unknown/duplicate IDs, and reports false-positive facts, false-negative facts, aggregate errors, and
citation errors. Quality passes only when all are zero.

The built-in blind-oracle selfcheck passed all 680 held-out records with zero errors. That validates
the grader wiring; it is not a candidate score.

## Leakage controls

The artifact verifier proves:

- public JSON preserves source `state`, `labels`, and closure time so exact baselines see the real payload;
- 30 private-boundary matches across eight records are replaced by a stable public-corpus marker before hashing/splitting;
- public JSON omits derived references, duplicate targets, contradiction cues, and aggregate answers;
- answer-bundle digest is absent from public JSON;
- train/tune/test IDs are disjoint;
- public and answer files bind to each other by SHA-256;
- the blind oracle traverses the same external submission/grader contract.

Operationally, a benchmark runner must mount/pass only the public corpus to candidate pipelines and
keep the answer path in the grader process. The files are both committed for reproducibility, so this
is **evaluation-process isolation, not cryptographic secrecy from a malicious process with repository
read access**. A future competition-grade run should place held-out answers in a separately
permissioned service or release them only after evaluation.

## Reproduce and verify

Source acquisition used the public GitHub CLI/API:

```bash
gh issue list --state all --limit 1000 \
  --json number,title,body,state,labels,createdAt,updatedAt,closedAt,url \
  > /tmp/fak-issues-1000.json

go run ./cmd/microcontextdemo \
  -corpus-input /tmp/fak-issues-1000.json \
  -corpus-public-output experiments/microcontext/s8f-github-issues-public-2026-08-09.json \
  -corpus-answers-output experiments/microcontext/s8f-github-issues-answers-2026-08-09.json \
  -corpus-report-output experiments/microcontext/s8f-github-issues-report-2026-08-09.json \
  -corpus-source 'github.com/anthony-chaudhary/fak/issues?state=all&limit=1000'
```

The live query is provenance, not a byte-stable replay after issues are edited. The committed public
snapshot and its digest are the immutable benchmark input. Verify that snapshot and its paired
answers/report with:

```bash
go run ./cmd/microcontextdemo \
  -verify-corpus-public experiments/microcontext/s8f-github-issues-public-2026-08-09.json \
  -verify-corpus-answers experiments/microcontext/s8f-github-issues-answers-2026-08-09.json \
  -verify-corpus-report experiments/microcontext/s8f-github-issues-report-2026-08-09.json
```

Grade a candidate without exposing answers to the candidate invocation:

```bash
go run ./cmd/microcontextdemo \
  -grade-corpus-answers experiments/microcontext/s8f-github-issues-answers-2026-08-09.json \
  -grade-corpus-submission /tmp/candidate-submission.json \
  -grade-corpus-output /tmp/candidate-grade.json
```

## Limits and next use

- Repository issue prose reflects this project's issue-writing conventions and is not representative
  of every large-input domain.
- State and labels are exact external metadata; relation and cue facts are deterministic text-derived
  labels. A later semantic benchmark needs human or separately modeled adjudication with agreement
  measurements.
- GitHub issue edits after the freeze do not alter the committed snapshot.
- Public issue numbers permit external lookups; live benchmark runners must deny candidate network
  access or regard such lookups as tool calls and charge/report them.
- The class distribution is natural rather than balanced. Reports must preserve slices instead of
  hiding weak classes in one average.

#6109 can now tune SQL/search, retrieval/rerank, long-context, and chunk configurations on train/tune
without inspecting the 680 held-out answers. #6110 then runs live endpoints; #6111 computes the
net-true decision boundary.
