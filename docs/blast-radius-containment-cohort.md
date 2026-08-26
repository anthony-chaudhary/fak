---
title: "Blast-radius containment - ticket cohort"
description: "The cohort of blast-radius containment tickets (#2712-#2720) filed on the fak repo, grouped so the containment work is tracked as a single unit."
---

# Blast-radius containment — ticket cohort

**Status:** already filed on `anthony-chaudhary/fak` as issues **#2712–#2720** (2026-07-05).
This page plus the nine ticket files beside it in `docs/blast-radius-containment/` are the
portable, self-contained spec, so another agent/fleet can (a) recreate the cohort in another
repo, or (b) read the whole design without touching the API. Copy the page and that folder
together and nothing is missing.

**Shape:** one epic (#2712) + one load-bearing spine (W1 / #2713) + seven fan-out children
(W2–W8 / #2714–#2720).

**Ship order:** W1 first (everything reads its ledger) → W2/W3 (recognize + estimate) →
the load-bearing W4/W5/W6 (hold + elect + release) → W7/W8 (surface + bound).

## The problem in one line
With N concurrent agents on a shared trunk, one agent's discovery of a *shared-root-cause*
failure silently taxes the other N-1: they **rediscover** it (burning a cycle each),
**collide** on the fix (N racing PRs to one file), or **stall globally** even though disjoint
work is shippable. Today's containment is all siloed — `guardrsi/livelock.go` is per-trace,
`attemptbudget` is per-issue, `blockerpost` is a human Slack post, `affectedtests` is
test-selection-only — so none of it recognizes "shared." This cohort makes the fleet discover
the shared failure **once**, hold **only** the affected agents, elect **one** fixer, and
**auto-release** on a *witnessed* fix.

## How to create these tickets
Each ticket file listed below carries a metadata block (title, labels, milestone, parent,
depends-on) followed by the issue body verbatim, so the file is filed as-is. To file one:

    gh api repos/<OWNER>/<REPO>/issues \
      -f title="<title>" \
      -F body=@<body-file>.md \
      -f 'labels[]=<label>' -f 'labels[]=<label>' ... \
      -F milestone=<N>

Notes (fak host specifics):
- Prefer `gh api` over `gh issue create` — the latter spelling can trip the guard.
- Labels and the milestone must already exist in the target repo. Milestone `6` here =
  "10x agentic coding loop with witnessed self-correction" (fak-specific; drop or remap
  `-F milestone` elsewhere).
- An outward-facing `gh api` write may hit the preview-confirm gate: re-propose the
  *byte-identical* call with `"_fak_confirm":"<token>"` added to the tool input. That is a
  pause, not a denial.
- Bodies use `##` section headers (Current state / Working spine / In scope / Out of scope /
  Done condition / Witness / Acceptance gate / Closure binding / Likely files / Lane / …) so
  each leaf passes the fak issue-contract dispatchability check.

## Cohort map
| Ticket | # | Seam | Package / verb |
|--------|---|------|----------------|
| Epic | 2712 | umbrella: recognize → broadcast → scope-hold → auto-release | — |
| W1 spine | 2713 | fleet-wide known-bad ledger + record/match | `internal/knownbad` |
| W2 | 2714 | recognize: cross-trace `FailureHash` correlation | `internal/guardrsi` |
| W3 | 2715 | estimate: blast radius = import graph ∩ live leases | `internal/blastradius` |
| W4 | 2716 | scope-hold only intersecting issues (`BLOCKED_BY_KNOWN_BAD`) | `internal/dispatchtick` |
| W5 | 2717 | elect exactly one fixer via an exclusive lease | `fak knownbad claim` |
| W6 | 2718 | witness-gated auto-release of held agents | `fak knownbad resolve` |
| W7 | 2719 | operator blast card (1 cause → N affected, 1 fixing) | `internal/blockerpost` |
| W8 | 2720 | TTL + revoke so a stale known-bad can't wedge the fleet | `internal/knownbad` |

Load-bearing for epic closure: **W1 + W4 + W5 + W6**.

## The nine ticket bodies

Each ticket lives in its own file under `docs/blast-radius-containment/`, carrying its metadata block and its issue body verbatim — so a body is filed with `-F body=@<file>.md` and read on its own without scrolling past the other eight.

| Ticket | # | File | What it lands |
|--------|---|------|----------------|
| Epic | 2712 | [`epic-2712.md`](notes/epic-2712.md) | umbrella: recognize the shared failure once, broadcast it, hold only the affected agents, auto-release on a witnessed fix |
| W1 spine | 2713 | [`w1-spine-2713.md`](notes/w1-spine-2713.md) | fleet-wide known-bad ledger with record and match (`internal/knownbad`) - the ledger every other ticket reads |
| W2 recognize | 2714 | [`w2-recognize-2714.md`](notes/w2-recognize-2714.md) | cross-trace `FailureHash` correlation so a shared root cause is recognized once (`internal/guardrsi`) |
| W3 estimate | 2715 | [`w3-estimate-2715.md`](notes/w3-estimate-2715.md) | blast radius as the import graph intersected with live leases (`internal/blastradius`) |
| W4 scope-hold | 2716 | [`w4-scope-hold-2716.md`](notes/w4-scope-hold-2716.md) | hold only the intersecting issues, with the `BLOCKED_BY_KNOWN_BAD` reason (`internal/dispatchtick`) |
| W5 elect fixer | 2717 | [`w5-elect-fixer-2717.md`](notes/w5-elect-fixer-2717.md) | elect exactly one fixer through an exclusive lease (`fak knownbad claim`) |
| W6 auto-release | 2718 | [`w6-auto-release-2718.md`](notes/w6-auto-release-2718.md) | witness-gated auto-release of the held agents (`fak knownbad resolve`) |
| W7 surface | 2719 | [`w7-surface-2719.md`](notes/w7-surface-2719.md) | the operator blast card: one cause, N affected, one fixing (`internal/blockerpost`) |
| W8 bound | 2720 | [`w8-bound-2720.md`](notes/w8-bound-2720.md) | TTL and revoke so a stale known-bad entry cannot wedge the fleet (`internal/knownbad`) |

Read them in ship order: W1 first (everything reads its ledger), then W2/W3, then the load-bearing W4/W5/W6, then W7/W8.
