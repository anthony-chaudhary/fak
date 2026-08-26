---
title: "Development tooling boundary: portable patterns versus fak machinery"
description: "Status: accepted for the fak / fak-dev split (6019, 6028)."
---
# Development tooling boundary: portable patterns versus fak machinery

Status: **accepted for the `fak` / `fak-dev` split** (#6019, #6028).

The runtime/dev split answers **which artifact executes a command**. It does not
answer whether the development idea is useful outside this repository. Those
are separate axes, and `fak index ownership --json` records both.

## The two-axis decision

| Runtime owner | Development reuse | Meaning | Intended audience |
|---|---|---|---|
| `runtime` or `shared` | `not-applicable` | Production kernel behavior. | adopters/operators |
| `dev` | `portable-pattern` | The concept is reusable, although today's command may still assume fak's repository. | examples, dogfood, possible product extraction |
| `dev` | `fak-maintainer` | Maintains, measures, documents, tests, or releases fak itself. | fak contributors |
| `dev` | `lab-operations` | Operates fak's private compute, accounts, or benchmark lab. | fak maintainers with lab access |

This classification is deliberately not named “public/private.” A portable
pattern can be implemented by internal code today; a fak-specific command can
use generic libraries. The classification describes the command contract and
why it exists, not where its Go package happens to live.

## Portable concepts we should preserve and teach

The initial inventory marks these families as portable:

- **Shared-tree coordination:** scoped commits, ownership-aware sweeps,
  peer-dirty-safe build checks, explicit-path validation, and detached worker
  worktrees. The reusable lesson is to name ownership and isolate evidence
  instead of pretending a concurrent checkout is clean.
- **Leases and typed blockers:** reserve a lane/resource before mutation,
  expire stale ownership safely, and return a typed reason when work cannot
  proceed. A current lease may be embedded under `worktree`, `dispatch`, or
  another command rather than exposed as one universal verb.
- **Contract-driven planning and dispatch:** shape issues/tasks into checkable
  contracts, dispatch bounded work, and make completion depend on a witness
  rather than worker narration.
- **Clean evidence:** distinguish committed-tip CI, “tip plus my named paths,”
  and the literal working tree. That distinction applies to any repository
  with concurrent agents.
- **Enforcement at the seam:** hooks can turn those conventions into refusals
  rather than prompt-only advice.

These are good candidates for adopter examples and dogfood because they are
work patterns, not claims that every adopter should reproduce fak's exact
trunk workflow.

## What stays fak-specific

Examples include release assembly, generated scorecards, repository concept
migration, fak's own claims and maturity ledgers, and compatibility audits.
They may contain useful implementation techniques, but their command contract
exists to evolve this codebase. They belong in `fak-dev`, not in the shipped
runtime and not in an adopter quickstart.

Lab operations are stricter still: fleet accounts, private nodes, scheduled lab
runs, and hardware-control commands depend on fak infrastructure and access
boundaries. Public docs may describe scrubbed evidence, but extraction must not
turn those controls into a general product surface.

## Concept portability is not API stability

`portable-pattern` is an invitation to **teach and extract**, not a support
promise for the current spelling, flags, output schema, or repository layout.
Before a pattern becomes adopter-facing, it must cross this promotion gate:

1. State the repository-neutral invariant without referring to fak's lanes,
   labels, hosts, branch policy, or issue taxonomy.
2. Put pure policy/state-transition logic in a package that does not import fak
   repository configuration or lab packages.
3. Provide a fixture or example repository that is not fak and capture an
   end-to-end witness there.
4. Define the public compatibility, security, and failure contract explicitly.
5. Only then expose it through runtime `fak`, a documented optional artifact,
   or an example. Until that point, `fak-dev` remains maintainer tooling.

A useful portable idea can therefore remain dev-owned indefinitely. Conversely,
moving a command to `fak-dev` does not make it suitable for users.

## Machine-readable contract

`fak index ownership --json` emits, for every top-level command:

- `owner`, `compatibility_name`, and `dispatch_target` for binary ownership;
- `dev_reuse` and `dev_reuse_rationale` for reuse intent.

Validation rejects a dev command with `not-applicable`, a runtime command with a
dev reuse class, an unknown class, or a missing rationale. New commands must be
classified as part of the same change.

There is intentionally **no fallback class**. When adding a top-level development
command, its dispatch/tier registration must be accompanied by exactly one edit
in `internal/devindex/devreuse.go`:

- `portableDevPatterns` with a command-specific repository-neutral rationale;
- `maintainerDevCommands` for an explicitly fak-bound command; or
- `labDevCommands` with a command-specific infrastructure rationale.

An absent or duplicate entry fails the exhaustive ownership test and names this
edit point. This is the default-enforcement seam for new commands, including
commands dispatched only by `fak-dev`; `OwnershipVerbs` joins extracted TierDev
rows back into the source-derived inventory. Representative and negative
classifications are pinned by tests in `internal/devindex/ownership_test.go`.

The inventory is the audit mechanism; this note is the interpretation contract.
Neither replaces the extraction and packaging work tracked by #6021–#6026.
