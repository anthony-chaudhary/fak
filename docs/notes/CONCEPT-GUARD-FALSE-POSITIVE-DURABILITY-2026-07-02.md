---
title: "Durable handling of guard false positives: match on structure, not mention"
description: "The reversibility classifier escalates commands that merely MENTION a trigger word — and is evadable by wrapping the same command in a script. Both are the one antipattern: matching on substring, not call structure. A durable design: the structural-match doctrine, a growing false-positive/true-positive corpus with a measured accuracy rate, and a logged escape hatch so a routed-around escalation always leaves a trace."
date: 2026-07-02
---

# Durable handling of guard false positives

Status: concept note + one shipped fix + one dispatched fix. The immediate bug is
[#2376](https://github.com/anthony-chaudhary/fak/issues/2376) (the reversibility
classifier); the companion in-band-confirm fix shipped as `29b98f26`; this note is
the durable design so the *class* of failure stops recurring, not just the instance.

## The episode (witnessed this session)

The #2156 reversibility preview gate classifies a tool call as
irreversible/outward-facing when trigger words appear **anywhere** in the command's
token stream, in order (`orderedWords`/`containsWord` over `commandWords(cmd)` in
`internal/adjudicator/reversibility.go`). On a guard-fronted session that develops
*this* repo — which talks about `git push`, `mail`, `rm`, `npm publish` constantly —
it self-collides. Five library-witnessed misfires (`go test -overlay`, at `29b98f26`):

```
outward-facing  grep -rn "git push" docs/
outward-facing  git commit -m "docs: explain when to push"
outward-facing  grep -c mail internal/gateway/*.go
irreversible    echo "never run rm -rf blindly"
outward-facing  git log --grep "npm publish"
```

Two more were first-person, live in this very session:

- A **pure-local hash computation** — `go run tok.go outward-facing PowerShell
  '{"command":"git push origin main",...}'`, which touches nothing outward — was
  escalated `REQUIRE_WITNESS/ESCALATE` because its *argument string* contained
  `git push`. The classifier escalated a command whose only sin was quoting the
  thing it was reasoning about.
- To file #2376 I wrapped `gh issue create` in a one-line script and ran
  `& mkissue.ps1`. The wrapper command carries no trigger word, so it sailed
  through — the **same matcher that over-blocks a mention under-blocks a wrapper.**

That symmetry is the whole point.

## The problem class: mention is neither necessary nor sufficient

A substring/subsequence match over the full command is wrong in *both* directions
at once:

- **Over-inclusive (false positive).** Any command that names the trigger as data —
  a grep pattern, a commit message, an `echo`, a `--grep` filter, a JSON arg — trips
  it. Cost: dead-ends on the content wire (before `29b98f26` the refusal rendered as
  a bare `Tool (/ESCALATE)` with no preview and no confirm token — unrecoverable),
  wasted propose→refuse→re-propose turns, and operator friction that scales with how
  much the repo *discusses* dangerous commands.
- **Under-inclusive (false negative).** The real effect lives in the command's
  *structure* — the executable at the head of each pipeline segment — not in whether
  a word appears. Wrap it (`bash -c "git push"`, `& script.ps1`, `xargs`), and the
  trigger word moves off the surface the matcher scans, so a genuinely outward call
  slips by.

A gate that a quoted string can fool in either direction is a **heuristic over text**,
not a **decision over structure**. The security-floor framing the repo already uses
(a good call can't silently slip below its budget; a bad call can't get through)
demands the classifier decide on what the shell will *execute*, not what the bytes
*mention*.

`curlWrites` already gets this right: its regex anchors `curl` to a segment boundary
`(^|[;&|()]|[[:space:]])`. It is the model the rest of the family should follow.

## The durable design — three layers

### 1. The doctrine: classify on call structure, not payload mention

State it as a checkable rule, the dual of the capability floor:

> A guard classifier matches on the **structure of the call it adjudicates** — the
> command head of each executed pipeline segment (env/wrapper prefixes stripped), the
> resolved write target, the actual egress host — never on the **mention** of a token
> anywhere in a payload it does not itself execute.
>
> Deliberate exceptions are *payload inspections* and must be named as such: SQL
> passed to a client (`drop database`), a `dd` target (`of=/dev/…`). These scan
> arguments **on purpose** and are the whitelist, not the rule.

#2376 is the first application: head-anchor the git/gh/npm/docker/mail/rm families
per segment; keep the SQL/dd scans as declared payload inspections. The same audit
should sweep the sibling classifiers (`commandOutwardFacing`, `commandIrreversible`,
egress host matching, the self-modify glob matcher) for the same substring shape.

### 2. The corpus: a labeled false/true set with a measured accuracy rate

Every misfire found in the wild becomes a permanent fence, so a fix is a ratchet, not
a patch. A single data file of `(command, expected_class, rationale)` rows —
seeded with the five #2376 mentions and the two live self-collisions above, plus the
true-positives that MUST still escalate (`git push origin main`, `rm -rf build`,
`git commit && git push`, `echo hi | mail bob`, `psql -c "drop database x"`). The
classifier test ranges over the corpus; a new false positive is *added*, never just
locally patched.

This is the net-true-value / observer-effect discipline pointed at the guard's own
accuracy: the classifier has a **false-positive rate** and a **false-negative rate**
against a labeled corpus, and those rates are a scorecard the garden/scorecard family
already knows how to fold and ratchet. "The guard is well-tuned" becomes a number
with a witness, not a vibe. Cheapest next step: land the corpus as the table-driven
body of `reversibility_test.go` under #2376, then lift it to a shared `testdata`
file the accuracy scorecard reads.

### 3. The escape hatch that leaves a trace (no silent workaround)

The failure mode that keeps this class alive is that a routed-around escalation is
*invisible*. This session's script-wrapper evasion taught the fleet nothing —
nobody learns the classifier over-fired. A durable surface fixes that:

- **In-band confirm recipe — shipped (`29b98f26`).** A true positive is now a
  one-turn pause, not a dead end: the refusal note carries the preview, the
  `_fak_confirm` token, and the dry-run hint, so a content-wire client can complete
  the two-phase confirm. This is load-bearing for the whole design: *the cheaper it
  is to confirm a real positive, the more aggressive the classifier may safely be* —
  over-blocking and confirm-cost must be co-designed, not tuned in isolation.
- **A logged false-positive report (proposed).** When an escalation is
  confirmed-innocent (by the operator, or by an authorized agent), record it —
  command, classifier, reason — to a ledger that (a) feeds the corpus in layer 2 and
  (b) surfaces the false-positive *rate* over time. A `fak guard false-positive`
  verb, or an entry the existing guard-audit journal already has a slot for. The
  rule mirrors "no silent caps": if you route around a guard decision, the routing
  must leave evidence, or the guard never improves.

## Second-order instance: the coarse hard-self glob

The same mention-vs-structure shape recurs one layer up. `internal/adjudicator/**`
is `hard-self` (`CORE_SELF_MODIFY`) *wholesale*, so fixing the reversibility
**preview** classifier — which is not the admission/witness/ship-grade machinery the
lock exists to protect — trips the core-lock and needs the maintenance-witness path.
The glob matches by **directory**, not by whether a file is truly coherence-bearing.
That is defensible as a conservative default (the maintenance-witness speed bump is
cheap and audited), but it is worth naming: if hard-self routinely fires on adjacent
non-self-grading files, the class boundary is matching location, not structure — the
same lesson at the corelock altitude. Not an action item here; a watch item for
`internal/corelocks`.

## How this ties to the loop-family work

The guard is one of the invariant *threads* (trust/witness) that
[recurs at every ring](../explainers/engineering-is-building-loops.md). Its
accuracy is an observability signal that belongs in a **loop**, not in an agent
stubbing its toe: the false-positive rate from layer 2 is exactly the kind of
re-derived-from-disk debt number the scorecard-control-pane and garden already fold
worst-first. Wiring `guard-accuracy` in as a scorecard member closes the gap between
"a guard misfired" and "the fleet measured and retired guard debt."

## Honest scope

- Shipped: the in-band confirm recipe (`29b98f26`, adversarially reviewed — hard
  denies win before the reversibility rung; the token was already on the MCP/HTTP
  wires; it is deterministic and model-derivable, a proof-of-deliberateness echo,
  not a capability).
- Dispatched: the #2376 head-anchoring fix, to a fresh headless session via the
  maintenance-witness path (hard-self; operator-authorized). A launch is not a ship
  — #2376 resolves only on a witnessed trunk commit.
- Proposed (cheapest next step named inline): the corpus-as-test → shared-testdata
  → `guard-accuracy` scorecard, and the logged false-positive escape hatch. None of
  these is shipped by this note.
- The two live self-collisions are first-person session evidence, not a captured
  transcript artifact; the five #2376 rows are library-witnessed at `29b98f26`.
