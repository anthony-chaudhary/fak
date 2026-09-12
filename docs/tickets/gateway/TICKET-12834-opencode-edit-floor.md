## Current state

The standard serving floor has exact-name allow entries for `Bash`, `Edit`, and `Write`, while OpenCode advertises lowercase `bash`, `edit`, and `write`. A pre-execution serving probe confirms ordinary lowercase `edit` and `write` reach `DEFAULT_DENY`; a read-only `bash` command is already admitted by the safe-command rung, while write-shaped lowercase shell actions still lack the explicit parity granted to `Bash`.

## Parent context

Related to #10824, which added OpenCode aliases to the embedded guard policy. The serving path uses the separate adjudicator default policy.

## Why this is next

A normal read now succeeds in the one-touch coding workflow, but the following edit cannot cross the serving floor, leaving the requested source change incomplete.

## Scope

Add the exact lowercase `bash`, `edit`, and `write` aliases to the standard capability floor and prove existing path checks remain authoritative.

## Working spine

1. Extend the default adjudicator allow map with the exact lowercase `bash`, `edit`, and `write` spellings.
2. Exercise `DefaultPolicy` through the existing adjudicator decision API.
3. Normalize path separators only for protected-fragment matching.
4. Verify ordinary arguments are preserved while protected paths remain denied.

## Core through-line

The serving gateway passes scoped lowercase OpenCode coding proposals to the local guard while retaining the adjudicator's existing pre-allow defenses.

## Gold-plating boundary

Do not add wildcard or case-insensitive tool admission, other tool aliases, argument normalization, gateway changes, or deployment behavior.

## Likely files

- `internal/adjudicator/default_policy.go`
- `internal/adjudicator/decide.go`
- `internal/adjudicator/*_test.go`

## Acceptance gate

- Lowercase `bash`, `edit`, and `write` are admitted for ordinary coding operations.
- Argument bytes are unchanged.
- Protected Linux and Windows paths remain denied.
- An unknown write-shaped tool remains default-denied.

## Done condition

The standard serving floor supports the three scoped OpenCode coding spellings without weakening path or unknown-tool denials.

## Witness

`go test ./internal/adjudicator -run OpenCode -count=1`

## Closure binding

Close only when the production change and independently authored regression are committed together and the focused witness exits 0.

## Lane

`adjudicator`

## Scope class

S0 single-package compatibility fix.

## Definition of done

The focused adjudicator test exits 0 with ordinary lowercase bash/edit/write admission, exact argument preservation, protected-path denial on both path styles, and unknown write-tool denial.

- [ ] Lowercase `bash`, `edit`, and `write` are admitted by `DefaultPolicy` for ordinary coding operations.
- [ ] Linux and Windows protected paths remain denied.
- [ ] Arguments are byte-preserved and an unknown write-shaped tool remains denied.

## Problem frame

- Centrality: Core (one-touch local coding task completion)
- P1 Context: preserved - no context or prompt behavior changes
- P2 Net value: advanced - the already-generated scoped edit reaches execution
- P3 Adaptation: preserved - tool arguments and downstream policy remain authoritative
- P4 Operations: advanced - the shipped serving path admits the client dialect it advertises




## GitHub issue

https://github.com/anthony-chaudhary/fak/issues/12834

