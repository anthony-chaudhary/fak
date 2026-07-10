# conceptbench task corpus — INDEX

The versioned corpus of **fak-concept tasks** for the concept benchmark
(#2730, epic #2721). Each task is a self-contained, fixture-backed episode: a
prompt, a reproducible starting state, and the **exact referee** the grader
(#2732) must read to score it — never the model's own "done" text.

The schema, loader, and hermetic fixture builder live in
[`internal/conceptbench/task.go`](../../../../internal/conceptbench/task.go);
the validating + golden-rebuild tests are in
[`task_test.go`](../../../../internal/conceptbench/task_test.go).

## Schemas

- **`fak.conceptbench.task.v1`** — `{schema, id, concept, prompt, fixture_ref,
  expected_witness, difficulty, notes}`. `concept ∈ {commit_stamp, lane,
  refusal, verdict_repair, hook_protocol, honesty}`. `expected_witness` MUST
  name one of the grader's known referees **and** be the referee that grades
  that concept (`internal/conceptbench/grade.go`) — the loader rejects any other
  with a typed `*TaskError` (`unknown_witness` / `witness_concept_mismatch`).
- **`fak.conceptbench.fixture.v1`** — a declarative, hermetic build recipe:
  `{schema, id, files?, git?, inject?}`. It materializes byte-identically across
  builds with no network, GPU, or key. `git` seeds a temp repo with fully pinned
  identity + dates (so HEAD SHA is deterministic); `inject` writes the non-file
  scratch state (a returned `verdict`, a `refusal_token`, live `leases`) under
  `inject/`.

## Concept → referee (the anti-masquerade map)

| concept          | expected_witness (referee)     | fixture kind        |
| ---------------- | ------------------------------ | ------------------- |
| `commit_stamp`   | `dos_commit_audit` / `dos_verify` | pinned git repo  |
| `honesty`        | `dos_commit_audit`             | pinned git repo     |
| `lane`           | `dos_arbitrate`                | injected leases     |
| `refusal`        | `dos_check_reason`             | injected token      |
| `verdict_repair` | `mcp.go:toolDescriptors()`     | injected verdict    |
| `hook_protocol`  | `fak.task-handoff.v1`          | clean-stop state    |

## Contents

- `fixtures/` — 6 fixture recipes, one per concept (`*.fixture.json`).
- `*.task.json` — 12 tasks, **2 per concept** (`<concept>-NN.task.json`).

## Status

- Loader + fixture rebuild are exercised by committed tests; a malformed task is
  rejected with a typed error (`internal/conceptbench` package tests).
- Golden proof: `TestFixtureGolden_ByteIdentical_*` and
  `TestOnDiskCorpus_FixturesRebuild` rebuild each fixture twice and assert an
  identical content `Manifest` (per-file sha256 + deterministic HEAD SHA).

## Verify

```
go test ./internal/conceptbench -run 'Task|Fixture|Corpus' -count=1
```

Related: grader adapter #2732, spine #2729, report #2739, scenarios #2733–#2737.
