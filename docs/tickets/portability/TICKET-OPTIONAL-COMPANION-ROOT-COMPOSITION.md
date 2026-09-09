---
lane: portability
paths:
  - pkg/repocompose/**
expected_steps: 6
work_unit: leaf
parent_ref: fak#11299
priority: P1
---

<!-- fak-portability-key: optional-companion-root-composition -->

# Compose optional companion roots without assuming an overlay checkout

## Current state

Public `fak` commands independently reconstruct companion-repository defaults. The
implementations disagree about whether an absent sibling is optional, fatal, or a
  directory that may be created. A recent `issuesolved` Git-mode repair handled one
  missing sibling, but the fix is local and can also suppress errors for required roots.

## Problem frame

- Repository selection feeds public CLI reports, multi-root discovery, goal
  synchronization, development tooling, and companion integrations.
- An implicit missing companion is the normal standalone topology; current defaults
  can fail, probe an inaccessible repository, or create a misleading sibling directory.
- fak#11299 already requests companion-root discovery for
  self-query. This leaf supplies its reusable typed input contract rather than
  duplicating the gateway integration.

## Value

- Centrality: Enabling (standalone public CLI and reusable repository discovery)
- P1: Advanced - one typed plan makes the normal standalone topology a first-class default.
- P2: Preserved - the probe is bounded local filesystem work with no network or cloning cost.
- P3: Advanced - ordered candidates let later consumers add overlays without changing the baseline.
- P4: Preserved - identity checks and required mode keep explicit operator intent fail-closed.

## Core through-line

Required public member plus ordered optional overlay candidates -> identity-checked
filesystem probe -> selected/skipped composition plan -> deterministic receipt that a
consumer can apply without guessing whether absence was expected.

## Parent context

fak#11299 requests companion-root discovery for multi-root self-query; this leaf
extracts the reusable root-plan contract needed before that gateway integration.

## Why this is next

The recent one-command repair proves the user-visible failure and the repository now
has multiple conflicting local fixes. A typed plan prevents the next consumer patch
from repeating or broadening the error-swallowing behavior.

## Gold-plating boundary

Do not clone repositories, scan arbitrary siblings, load executable plugins, change
repo-guard authorization, alter harness-lock activation, or migrate a consumer in this
leaf. The contract only plans already declared local repository members.

## Definition of done

- [ ] Add `pkg/repocompose` with typed `auto`, `required`, and `off` policies plus a versioned receipt.
- [ ] Candidate precedence is caller-defined and stable; duplicate roots collapse.
- [ ] Auto absence is recorded without error, while required absence returns a typed actionable error.
- [ ] A present candidate with the wrong identity marker fails and is recorded as invalid.

## Witness

```text
go test ./pkg/repocompose -run TestCompose -count=1
```

The test uses synthetic repositories and proves public-only success, valid overlay
composition, deterministic precedence/deduplication, explicit absence failure, and
present-but-invalid failure.

## Likely files

- `pkg/repocompose/repocompose.go`
- `pkg/repocompose/repocompose_test.go`

## Lane

portability

## Expected steps

6

## Routing

```routing
lane: portability
paths:
  - pkg/repocompose/repocompose.go
  - pkg/repocompose/repocompose_test.go
expected_steps: 6
```

## Working spine

`repocompose.Compose(primary, overlays...)` produces a selected root list and a
versioned receipt; later consumer issues adapt `issuesolved`, self-query, and goal sync.

## Acceptance gate

The targeted test passes from a checkout that has no sibling overlay repository, and
the package imports only the Go standard library.

## Closure binding

Close only when the implementation commit contains both likely files and the exact
targeted witness passes. Follow-up consumer migrations remain separate issues.
