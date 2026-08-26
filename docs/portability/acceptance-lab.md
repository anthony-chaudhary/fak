---
title: "Portability acceptance lab (#6606)"
description: "go run ./cmd/portability-lab is the independent, credential-free release gate for epic 6589. It calls the shipped internal/portability APIs directly;"
---
# Portability acceptance lab (#6606)

`go run ./cmd/portability-lab` is the independent, credential-free release gate for epic #6589.
It calls the shipped `internal/portability` APIs directly; it does not duplicate their implementation.

```sh
go run ./cmd/portability-lab --out report.json
jq -e '.verdict=="pass" and .hermetic and .active_state=="none" and all(.coverage[]; .status=="proven")' report.json
```

The JSON is authoritative. `coverage` uses the closed enum `proven|partial|unsupported|untested`,
identifies the real API used by every witness, and includes all epic and #6606 acceptance IDs.
The process exits non-zero for a missing/non-proven requirement, external dependency, or ambiguous
active state. Failure-path evidence is persisted under the run's `failures/` directory before exit.

Metrics label provenance as `witnessed`, `observed`, or `modeled`; each row names its tuned
alternative, baseline value, and net cost. The baseline is the tuned manual export/import checklist,
not a naive no-tool comparison. CPU is explicitly an upper-bound model; correctness, behavior,
leaks, decisions, bytes, and expert controls are witnessed from the run.

The `active-loop-history-checkpoint` scenario is the portability integration seam for #6432: a
paused active-loop object and history checkpoint move between isolated homes atomically and without
duplication. No process orchestration, credential, network service, or private fixture is required.
