---
title: "Complain by default — the agentic-dev friction channels and when to use each"
description: "Friction that no agent records never gets fixed: the fleet's hard-won knowledge of annoying agentic-dev things stays trapped in per-agent private memory, invisible to the repo and to the next agent who hits the same wall. This note makes complaining a default: a closed taxonomy of the recurring friction classes, each routed to the structured, deduplicating channel that turns one agent's annoyance into a tracked, witnessed fix — plus how to complain so it is signal, not noise, and the honest current state of the channels."
---

# Complain by default

> Design note + playbook. Snapshot for `/goal` 2026-06-29 ("improve the default
> ability for agents to complain about annoying agentic-dev things"). Each channel
> is cited by package / verb; the one channel that is **in-flight today** is
> labelled `not yet` with the next checkable step, per the repo's `not yet`
> doctrine. Companion of the [agent programming grammar](CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md)
> and the durable [innovations index](../INNOVATIONS-INDEX.md).

## The thesis

The expensive failure mode of an agent fleet is not the agent that complains too
much. It is the agent that hits an annoying agentic-dev thing — a shared-tree
clobber, a half-landed RED trunk, a stamp gate that refuses an on-lane commit, a
push blocked on a token-shaped fixture — and **silently routes around it**. The
workaround restores that one agent's forward motion and records nothing, so the
wall is still there for the next agent, who pays the same tax and routes around it
again. Friction no one records never gets fixed; it just gets re-discovered.

The evidence is that this knowledge already exists — but in the **wrong place**.
Each agent accumulates a private memory of the fleet's papercuts (the trunk
guard's `OFF_TRUNK`, the commit-audit `ABSTAIN` traps, the `GOTOOLCHAIN=local`
escape for a saturated box, the `dos_*` MCP timeout under load). That store lives
under the agent's own home directory, unversioned and unreadable by peers. The
default response to friction should instead be to **register it through a
structured, deduplicating channel** that is part of the repo — so the second
occurrence escalates a single tracked record instead of re-teaching one private
memory at a time.

This is the same invariant the rest of the substrate carries (see the grammar
note): *a state no participant can move by narrating around it.* A complaint is
just friction lifted into a first-class, refusable, witnessed value — the same way
[`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) lifts a claim and `dos_refuse_reasons` lifts a
refusal. "I worked around it" is a self-report that moves nothing; a filed,
deduped, witnessed complaint is the thing a maintainer can actually fix.

## The friction taxonomy (the "what")

A closed, evidence-grounded set of the recurring annoyances. Each is a *class* (so
two agents hitting the same wall fold onto one escalating record), with the channel
that fits it. The classes are drawn from observed fleet friction, not invented.

| class | what it is | default channel |
|---|---|---|
| `shared-tree-clobber` | a peer session reverts/deletes your **uncommitted** edit on the shared trunk tree (commit-by-path sweeps, or a file you were editing vanishes mid-session) | recurring friction → deduped issue; the durable fix is per-session isolation |
| `red-trunk` | `make ci` is red on `main` because of **someone else's** half-landed work, so your green lane cannot land | operator blocker if it stalls the fleet; else background status |
| `stamp-refusal` | the commit stamp / `dos verify` referee refuses or `ABSTAIN`s a legitimate commit (off-lane trailer, a non-`Conventional-Commits` verb, an embedded parenthetical in the subject) | guard appeal (capability-floor sibling) → deduped issue |
| `push-blocked` | a push is blocked by secret push-protection on a token-shaped test fixture, or by the trunk guard (`OFF_TRUNK`, a peer merge in flight) | background status → operator if it blocks a release |
| `saturated-box` | `go`/`go test` crawls or is killed at a wall-clock timeout because peer builds saturate the box; the escape is a warm shared cache + `GOTOOLCHAIN=local` | background status (capacity), not a per-agent fault |
| `tool-timeout` | a tool an agent depends on times out under load (e.g. the `dos_*` MCP server; fall back to the `dos` CLI) | recurring friction → deduped issue |
| `lane-collision` | two agents contend for the same files/lane because the arbiter was not consulted first (`dos_arbitrate` would have refused one) | recurring friction → deduped issue |
| `guard-false-positive` | the capability floor refused a **legitimate** tool call — byte-identical in the journal to a correct refusal, so only the agent that made the call knows it was wrong | guard appeal, with the witnessed verdict attached |
| `confusing-refusal` | a gate/guard refused with a reason that did not tell the agent **how to recover** | guard appeal (`confusing`) → deduped issue |
| `phantom-artifact` | an issue or plan names a file/symbol that does not exist (a fabricated citation in the work itself) | verify with `dos_verify` / `dos_recall`, then file/close honestly |

The point of naming the closed set is that an agent in the middle of friction does
not have to *invent* a category under pressure. It picks the class, and the class
already knows where the complaint goes.

## The channels (the "where")

There is no single verb that swallows every complaint, because the channels differ
in **who acts and how fast**. Routing the complaint to the right one is what makes
it actionable instead of noise.

1. **Operator-needed impediment** — something a human must clear (host down,
   GPU-gated work waiting on GPU server hours, a release blocked). Channel: **`fak blockers
   post --severity operator`** (`cmd/fak/blockers.go`, `internal/blockerpost`). This
   is the stable, shipped surface; `operator` pages `<!here>` (or a named owner),
   carries a "do this next" action, and folds the open backlog into one roll-up via
   `fak blockers feed`. Severity is **surfacing, not volume**: `status` is a quiet
   background record, `operator` pulls attention, `clear` is the all-clear heartbeat.

2. **Capability-floor / kernel-decision appeal** — the agent judges a `fak guard`
   refusal wrong (a false positive, an over-broad gate, a confusing reason). This is
   the **subjective** channel: a false-positive `DENY` is byte-identical to a correct
   one in the decision journal, so the objective guard RSI loop cannot surface it —
   only the agent that made the call knows. The complaint must carry the **witnessed
   verdict** pulled from the journal (a self-report is not a witness), and repeat
   appeals about the same class fold onto one escalating, deduplicated issue.
   `not yet`: the dedicated `fak complain` verb (backed by `internal/guardcomplaint`)
   is **in-flight on the shared tree** — the package exists but its CLI dispatch is
   not wired in `cmd/fak/main.go` today (the verb file was being reworked while this
   note was written). Until it lands, attach the journal `DENY` row as evidence and
   file the appeal through the dogfood-issues fold by hand. **Next checkable step:**
   once the complain surface settles, add the `case "complain"` dispatch + a
   `workflow`/`dev` complaint domain so the *same* deduping channel covers general
   agentic-dev friction (the classes above), and reference it from
   [`AGENTS.md`](https://github.com/anthony-chaudhary/fak/blob/main/AGENTS.md) so it is discoverable by default.

3. **Structured, verifiable refusal** — when the right move is to *decline green
   with a reason* rather than render an unearned OK. Channel: the DOS closed refusal
   vocabulary (`dos_refuse_reasons` to browse, `dos_check_reason` to validate before
   emitting). A refusal token is simultaneously emittable, verifiable, and refusable
   — that is what makes "no" a first-class value the loop can route, instead of
   free-text prose that drifts to `UNCLASSIFIED`. Use this when the complaint *is*
   the reason you are not proceeding.

4. **Recurring process friction worth a tracked fix** — the shared-tree classes,
   tool timeouts, lane collisions. Channel: a **deduped GitHub issue** (the
   `internal/dogfoodissues` fold that the guard-appeal path already reuses): a stable
   marker key folds the N-th occurrence onto one issue and bumps an occurrence count,
   so a recurring papercut becomes a *stronger* signal over time rather than a pile
   of duplicates.

## How to complain well (so it is signal, not noise)

A complaint channel earns its keep only if the complaints are good. Four rules,
each one a property the channels above already enforce:

- **Structured and deduped.** One class per complaint, with a stable key. The
  second occurrence must escalate one record, not open a second issue. Volume is not
  severity.
- **Carry a witness, not a feeling.** Attach the journal verdict, the failing `make
  ci` line, the git evidence — the non-forgeable record that the thing you are
  complaining about actually happened. An inferred cause stated as fact is the
  failure the no-unwitnessed-causal-claims rule exists to catch; label a cause
  inferred, or wire the witness first.
- **Name the recovery you wanted.** A `confusing-refusal` complaint that says only
  "this was confusing" is unactionable; one that says "the reason did not tell me to
  re-stamp with the on-lane trailer" is a fix.
- **Route by who acts.** Operator-paging for a human-needed clear; a deduped issue
  for a process fix; a refusal token for a green decline. Mis-routing a background
  papercut to `operator` is the fastest way to get the channel muted.

## Honest status

- **Shipped today:** `fak blockers post|feed` (operator/status/clear surfacing) and
  the DOS refusal vocabulary (`dos_refuse_reasons` / `dos_check_reason`) are the
  stable channels an agent can use right now. The dedup/escalation machinery
  (`internal/dogfoodissues`) is shipped and reused by the appeal path.
- **`not yet`:** a single, discoverable, low-friction verb that covers *general*
  agentic-dev friction (not just guard appeals) is not wired. The `guardcomplaint`
  appeal engine exists but its `fak complain` dispatch is mid-rework on the shared
  tree, and `cmd/fak/main.go` is peer-churning, so wiring it this pass would entangle
  peer WIP. The next checkable step is in channel (2) above: wire the dispatch and
  add the `workflow`/`dev` domain once the surface settles, then surface it in
  `AGENTS.md`. This note is the durable half — the taxonomy and routing an agent
  needs to complain well *today*, through the channels that already exist.
