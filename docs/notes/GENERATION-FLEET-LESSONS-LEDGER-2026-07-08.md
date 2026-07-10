---
title: "Generation classification: fleet lessons ledger (2141)"
description: "Generation-triage record classifying issue #2141 (cross-agent fleet lessons ledger) as gen/next, with the evidence and smallest next implementation step."
---

# Generation classification: fleet lessons ledger (#2141)

This note is the generation-triage record for
[#2141](https://github.com/anthony-chaudhary/fak/issues/2141)
(*feat(fleetmemory): cross-agent lessons ledger — publish a hard-won
recovery/scar once; auto-inject to peers who enter the same context*). It exists
because #2141 landed `unclassified`: labels `dev-ex` + `track/F-integration-tooling`,
milestone *10x agentic coding loop with witnessed self-correction*, but **no
`generation` stream label**. Per [`docs/generation.md`](../generation.md), an
`unclassified` issue is *held for classification repair* before it is
dispatchable in a generation view. This note supplies the horizon and the
evidence so a later worker can pick #2141 up without re-reading the parent epic
([#2136](https://github.com/anthony-chaudhary/fak/issues/2136)).

## Classification: `gen/next`

Stream label `gen/next`, generation milestone *Generation G1 - Next Gen*.

The issue asks for a fleet-wide ledger where one agent publishes a lesson once
with a **trigger context** (host, path glob, tool, refusal token), and any peer
whose session matches that trigger receives it *before* hitting the wall —
**read-time re-verified** so a stale lesson is withheld, not asserted.

Measured against the [`docs/generation.md`](../generation.md) stream table:

- **Not `gen/now`.** `gen/now` requires *no dependency on a future architecture
  bet*. This depends on a shared, arbitrated, provenance-carrying memory store
  that is not built yet. The shared store today is *unarbitrated, unverified, and
  forked* — see
  [`RESEARCH-DURABLE-SESSION-STATE-SHARED-MEMORY-2026-07-01.md`](RESEARCH-DURABLE-SESSION-STATE-SHARED-MEMORY-2026-07-01.md)
  §3–§4 (gap **G4**). A ledger with read-time re-verify sits *on top* of that
  missing composition, so it is not current-product-with-a-witness work.
- **`gen/next` — the fit.** Every primitive it needs already ships; what is
  missing is the composition plus a gate, a schema, and a dogfood witness —
  which is precisely the `gen/next` definition ("near-term foundation… still
  needs a gate, dogfood run, schema, or default-exposure proof"):
  - the auto-memory store (`.../memory/`, mirrored by `tools/sync_memory.py`,
    read by `tools/memory_read.py`) — the substrate a lesson is published into;
  - `dos_recall` — read-time re-verification of a recalled memory's claims
    against git + working tree (the "withhold a stale lesson" mechanism the issue
    names explicitly);
  - `dos_arbitrate` / lanes — multi-writer admission by disjoint file trees (the
    "publish once" without a last-write-wins race);
  - `internal/devindex` — the host/path/tool/refusal-token surface a trigger
    context would match against;
  - `tools/memory_cotravel.py` — the working **shadow→live gate + append-only,
    size-capped ledger** pattern (`FAK_MEMORY_COTRAVEL`, pluggable `STRATEGIES`)
    that a lessons ledger can reuse verbatim to prove value before it copies
    anything.
  This is the same shape as `RESEARCH-DURABLE-SESSION-STATE-…` §5 proposals
  **P4** (witnessed promotion gate) and **P5** (shared memory as an arbitrated
  lane): #2141 is the *trigger-context-indexed* instance of those bets.
- **Not `gen/second-next`.** That horizon needs simulation, a compatibility
  policy, or cross-generation dependency management first. Here the two field-
  hard parts — recall-time verification and multi-writer arbitration — are
  *already shipped mechanisms* (`dos_recall`, `dos_arbitrate`), so the work is
  nearer than second-next.

## Smallest next implementation increment (for the worker who picks this up)

Not built by this note. The first checkable rung:

1. Define a **lesson record schema**: the existing distilled fact plus a
   `trigger` block `{host?, path_glob?, tool?, refusal_token?}` and a provenance
   digest (source path + git blob) so `dos_recall` has a mechanical binding to
   re-verify at read time.
2. **Publish path**: promote a per-store lesson (e.g. the repo's own
   `bash_git_gh_hang_use_powershell`, `wsl_go_test_capture_technique`,
   `git_commit_flag_order_gotcha`) into the shared committed mirror carrying its
   trigger block.
3. **Match + inject at session start**: filter shared lessons whose trigger
   matches the starting session's `{host, cwd/path, available tools}`, run each
   surviving lesson through the `dos_recall` re-verify, inject the fresh ones.
4. **Gate it shadow-first**, mirroring `memory_cotravel.py`: `shadow` computes
   and ledgers which lessons *would* inject (measure hit-rate and false-trigger
   rate) and injects nothing; flip to `live` once the ledger shows the match is
   precise.

**Ratchet constraint for that worker:** do **not** add a new `tools/*.py` — the
pythongate ratchet (`go test ./internal/pythongate -run TestNoNewPythonTools`,
`REASON_NEW_PYTHON_TOOL`) reds on a new Python tool. Port the matcher/injector as
a `fak` subcommand (Go), or extend an existing tool.

## Milestone note (repair flagged, not silently applied)

`gen/next` should agree with the *Generation G1 - Next Gen* milestone
([`docs/generation.md`](../generation.md): a generation view requires both the
stream label and its matching milestone). #2141 currently carries the program
milestone *10x agentic coding loop with witnessed self-correction*, and GitHub
allows only one milestone per issue — so binding G1 would *drop* the program
milestone. This note applies the `generation` + `gen/next` **labels** (the
horizon is unambiguous) and flags the milestone rebind as an operator decision
rather than guessing, because dropping the program milestone is a lossy,
non-obvious side effect.

## Evidence ledger (generation-close contract)

- **Promotion evidence** (what would move #2141 toward `now`): the shadow gate
  above lands and its ledger shows lessons matching real peer sessions with a low
  false-trigger rate, and a fresh peer on a known-bad host receives the published
  lesson at start (the issue's own witness) without firing the wall.
- **Demotion / retirement evidence** (what would move it away): the trigger
  match proves imprecise (peers get lessons that do not apply) with no cheap
  precision fix; or `dos_recall` re-verify cannot bind because published lessons
  carry no stable provenance digest; or the shared store stays forked
  (`RESEARCH-DURABLE-…` §4 **G4** / §5 **P6**) so the verifier aims at the wrong
  root — any of these demotes the ledger until the underlying store work
  (P4/P5/P6) lands first.
- **Invalidating assumption**: this classification assumes a lesson's trigger is
  cheaply and *stably* computable from a session's start context
  (host + cwd/path + tool set + refusal token). If triggers turn out to need live
  behavioral signal that is only knowable *after* the wall is hit, "auto-inject
  before the peer hits the wall" is not achievable as stated and #2141 demotes to
  `gen/second-next` (needs a simulation/replay harness to learn triggers first).
