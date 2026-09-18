---
title: "Presence is not invokability"
description: "An artifact that exists in the tree is not thereby reachable from production: the static-presence vs production-reachability distinction, its evidence ladder, its named failure classes, and the one-command witness per surface."
---

# Presence is not invokability

An artifact that **exists** in the tree — a file, a symbol, a doc, a test, a claim — is not
thereby **reachable** from a production path. Presence is a static fact: the bytes are
committed at `HEAD`. Invokability is the reachability fact: some production entry point
actually calls the artifact at `HEAD`. A green unit test that never crosses the production
boundary proves presence, not invokability.

This distinction is the load-bearing one behind the repository's proof-by-default rule. A
fix that adds a symbol nobody calls has shipped no behavior. A doc that describes a runtime
path nobody invokes describes a path that does not exist. The public oracle that makes the
reachability claim checkable is
[`internal/architest/islands_audit_test.go`](../internal/architest/islands_audit_test.go):
`TestFeatureIslandsWiredAudit` (alias `TestIslandsAudit`) statically parses non-test `.go`
files and asserts that a named production function body actually calls a specific callee
symbol; a miss reports `ghost implementation island detected!`.

## Presence → invokability ladder

The ladder ascends from the cheapest static fact to the strongest behavioral one. Read it top
to bottom: each rung is strictly stronger, and a claim may climb higher but never skip a rung.

| Rung | What it establishes | What it does not |
|---|---|---|
| **0. Present** | The artifact is tracked and committed at `HEAD`. | Nothing about reachability or behavior. |
| **1. Reachable** | A production entry point calls the artifact at `HEAD` (the island audit's claim). | Nothing about whether the behavior is correct. |
| **2. Exercised** | The production path runs the artifact on-device and the run is captured. | Nothing beyond the captured envelope. |
| **3. Witnessed** | The behavior is observed on the real path and matches the claim. | Nothing outside the measured operating envelope. |

Rungs 0–1 are static; rungs 2–3 are behavioral. The minimum evidence on any rung is still a
checkable artifact — this ladder specializes the existing ladders rather than forking them:

- [`docs/generation-witness-ladders.md`](generation-witness-ladders.md) fixes the
  minimum-evidence rung per generation stream and carries the invariant that **no rung is
  ever "no witness."**
- [`docs/standards/verification-ladder-spec.md`](standards/verification-ladder-spec.md) is the
  declarable `G2` rung ladder; its cost set is closed
  (`reuse < in_process < corroborate < suite < worktree_spawn < human`) and its
  `on_exhaustion` is pinned to `deny`.

Neither schema is restated here; climb the presence→invokability ladder within whichever
rung ladder applies to the claim.

## Named failure classes

Presence-vs-invokability failures are named in public terms, never by forking a private
taxonomy. These are named failure classes, not a closed token set; the machine-checkable
form lives in the companion doctrine (a private issue, if cited, is qualified as
`fak-private#<n>`). The public classes are:

- **Ghost island** — a symbol or file exists at `HEAD` but no production entry point calls it.
  This is the class `TestFeatureIslandsWiredAudit` refuses.
- **Unwired witness** — a test passes without crossing the production boundary, so the real
  path is never exercised.
- **Stale description** — a doc describes a runtime path that no production caller invokes.

The vocabulary these classes report through is closed, not free text, and this page defers to
the canonical sources rather than restating them:

- [`internal/abi/reasons.go`](../internal/abi/reasons.go) is the closed adjudication
  refusal vocabulary (`CoreReasonCount`); see
  [`docs/faq/adjudication-lock.md`](faq/adjudication-lock.md) for the full list and meaning.
- [`docs/standards/agent-grammar.md`](standards/agent-grammar.md) states the general
  "closed vocabulary, evidence-bound, fail-closed" rule a conforming verb must keep.
- The DOS `[reasons.*]` set in `dos.toml` is the refusal token vocabulary; its recovery path
  is the `AGENTS.md` "If the kernel refuses you" section.

## One-command witness per surface

| Question | Start with | What a pass proves |
|---|---|---|
| Is the artifact reached from production? | `go test ./internal/architest/ -run TestFeatureIslandsWiredAudit -count=1` | Every audited production function actually calls its callee at `HEAD`; it does not prove behavior correctness. |
| Is a tracked Markdown doc reachable at `HEAD`? | `fak-dev index graph --json` | A HEAD-only census of tracked Markdown; it does not prove the runtime behavior of code the doc describes. |
| Does the change compile while hiding peer WIP? | `fak-dev buildcheck --vet --mine <path>...` | The isolated overlay compiles and vets the named paths; it is not the committed-tip gate. |
| Is the committed tip clean? | `fak-dev ci-preflight` | An archived checkout of `HEAD` is gofmt-clean and buildable without peer WIP. |

## Agent rule

Before claiming a fix is done, you **MUST** name the production entry point at `HEAD` that
reaches the changed artifact and run its one-command witness. You **MUST NOT** present a
passing test that never crosses the production boundary as proof of invokability — that is
presence. If no production entry point reaches the artifact, say `not yet` with the missing
reachability witness instead of claiming a fix.

## Self-check

- [ ] I named the production entry point at `HEAD` that calls the changed artifact.
- [ ] I ran the one-command witness for that surface and can quote its output.
- [ ] My passing test actually crosses the production boundary, not just an in-memory mock.
- [ ] Any doc I changed describes a path a production caller really invokes.
- [ ] If no reachability witness exists, I said `not yet` rather than claiming done.

## Related authorities

- [`AGENTS.md`](../AGENTS.md#proof-by-default-every-issue-fix-ships-its-evidence) — the
  public "Proof by default" section.
- [`docs/dev-tooling.md`](dev-tooling.md) — the sibling developer-tooling guide, including the
  proof-depth ladder and the HEAD-only documentation census.
- [`CLAIMS.md`](../CLAIMS.md) — the shipped-vs-stub ledger, claim by claim.