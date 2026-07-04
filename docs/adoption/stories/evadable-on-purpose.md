---
title: "Roughly 100% Evadable, On Purpose: the Detector Honesty Story"
description: "fak says its own injection detector is roughly 100% evadable, on purpose. The floor is the default-deny capability lock plus quarantine, not detection."
slug: evadable-on-purpose
keywords:
  - prompt injection
  - injection detector
  - evadable by design
  - default deny
  - capability floor
  - quarantine
  - containment
  - honest security
  - threat model
  - result admission
date: 2026-07-03
---

# Roughly 100% Evadable, On Purpose

Most security tools lead with what they catch. This one leads with what it does not.
fak ships a detector that flags suspicious tool results, and the repo says in plain
text that a determined attacker can get past it roughly every time. That admission is
not a disclaimer buried in a footnote. It is the point of the design, and it is what
makes the rest of the security story believable.

The reason the concession is safe to make is simple: the detector is not what keeps
you safe. Two other things do, and neither of them has to recognize an attack to stop
it.

## The claim, stated the way the repo states it

fak's result-admission gate screens every tool result before it can enter the model's
context. Secret-shaped and prompt-injection or poison-shaped results are quarantined,
held out of context and paged to a stub pointer, per the result-admit entry in
[CLAIMS.md](../../../CLAIMS.md). That gate is fed by a heuristic detector, and the
honesty about it is direct. CLAIMS.md records that the detector "is ~100% evadable on
a SOTA evasion battery and FP-prone on private real-transcript corpora," and that
"Detection is deliberately non-load-bearing." The security policy says the same thing
from the attacker's side: [SECURITY.md](../../../SECURITY.md) puts evading the detector
explicitly out of scope. The heuristic that flags suspicious tool results is, in its
words, "≈100% evadable by design," a "helpful bonus, never the floor," and
"A prompt that the detector doesn't flag is not a vulnerability."

So the number is not a marketing figure someone reverse-engineered. It is the repo's
own stated ceiling, written into the two documents an outside researcher would read
first.

## What actually holds when the detector misses

If the detector catches nothing, two structural gates remain. Both are described in
[Why default-deny beats a classifier](../../explainers/default-deny-vs-classifier.md)
and in [The tool call is a syscall](../../explainers/tool-call-is-a-syscall.md).

The first is structural containment. Quarantine does not depend on understanding the
attack. An untrusted tool result is held out of the model's context by default and
paged to a stub, so poisoned bytes never sit in the model's attention where they could
become its next instructions. The result is walled off by structure, not by a
successful catch. CLAIMS.md also records the durable version of this: a page the gate
sealed at write time is refused on page-in across a process boundary unless a witness
clearance ran and the bytes pass a fresh re-screen, so a quarantined result cannot
launder itself into a later session.

The second is the default-deny capability floor. Every tool call is checked against a
reviewable allow-list, and an action that was never allow-listed cannot run, no matter
what text the model was talked into emitting. There is no argument for an attacker to
win, because the dangerous lever was never wired up. SECURITY.md lists a
capability-floor bypass and a containment bypass as in-scope security bugs, right next
to the line that says beating the detector is not one. That split is the whole posture
in one place: the detector is optional, the floor and the containment are load-bearing.

## Why volunteering the weakness is the stronger move

A defense built on recognition has to win every time, forever, against an attacker who
can iterate phrasing offline until something slips through. A defense built on
structure does not have that shape. Refusing an action that was never granted, and
holding an untrusted result out of context, both work on the attack the detector
missed. The detector becoming evadable does not move the floor.

That is why the concession costs nothing. If the security depended on detection, saying
the detector is evadable would be an admission of failure. Because the security does
not depend on detection, saying so is just an accurate description of a bonus feature.
The honesty is load-bearing in a different way: it tells a careful reader exactly where
to attack, and the answer is the floor and the containment, not the screen.

## The honest fences on the fences

Two things this story does not claim.

The detector is not only evadable, it is also false-positive prone, and the repo shows
the receipts rather than hiding them. CLAIMS.md records a real session where the gate
sealed 2 of 59 pages, two large base64 image renders flagged as `SECRET_EXFIL` that
were benign, on the same evadable-and-FP-prone ceiling. Making the detector's decisions
durable and queryable does not make them better decisions.

And the floor is not a proof of total safety. The standing residual of any capability
floor, noted in the default-deny explainer, is that an injected prompt can still steer
a policy-allowed call with malicious arguments. Argument-level deny rules and egress
sink-gating reduce that, but do not erase it. The claim here is narrow and exact: the
security does not rest on catching the attack, so conceding that the catch is evadable
takes nothing away.

## The one line to keep

The detector is roughly 100% evadable, and fak says so in its own security policy,
because the floor that keeps you safe is the default-deny capability lock and the
structural quarantine, neither of which has to recognize the attack to stop it.

## See also

- [Why default-deny beats a classifier](../../explainers/default-deny-vs-classifier.md) — the same posture with one attack walked through both stacks.
- [The tool call is a syscall](../../explainers/tool-call-is-a-syscall.md) — the mental model the floor comes from.
- [What fak is not](../../explainers/what-fak-is-not.md) — the detector concession in the wider honest-boundary list.

_Dimension H (Benchmark-as-story) of the [concept-popularization epic](../../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md)._
