---
title: "The monotone-tighten self-amendment envelope: a provably-safe frontier for agent self-policy"
description: "Design spike for epic #5170 Track E. Defines the narrow envelope in which a wrapped agent may amend its OWN guard floor without ever granting itself a capability: (1) monotone-tighten self-writes, admitted only when the widening delta is provably empty, and (2) propose-only widenings the agent queues for out-of-band operator ratification. Specifies the file seams, the admission predicate, and the invariants for #5181 and #5182."
date: 2026-07-17
---

# The monotone-tighten self-amendment envelope

Status: design spike for epic #5170 Track E (the SELF-AMENDABLE frontier). It
specifies the two mechanisms that open a *provably-safe* self-amendment envelope
without ever letting an agent grant itself a capability. Implemented by #5181
(self-tighten admission) and #5182 (propose-only ratification).

## The rule that cannot bend

**An agent can never grant itself a capability.** Every widening channel
(GATED-WIDEN) is, by construction, one the wrapped agent cannot reach — the
overlay files are in `SelfModifyGlobs`, the reload endpoint is loopback + env
gated, operator escalation is a human. This is the load-bearing property of the
whole guard. The frontier work must not weaken it.

So "self-amendable" cannot mean "the agent widens its floor." It can only mean
two things that are provably safe:

1. **The agent tightens its own floor.** Strictly reducing your own capability is
   always safe — it can only ever refuse more. This is monotone-tighten.
2. **The agent asks.** The agent emits a structured *request* to widen that grants
   nothing until an out-of-band operator ratifies it. This is propose-only.

Both keep the invariant: nothing the agent writes, by itself, loosens the floor.

## Mechanism 1 — monotone-tighten self-write (#5181)

### Seam
A new optional overlay file `.fak/guard/self-tighten.json` that the *wrapped
agent is permitted to write* (i.e. it is NOT in `SelfModifyGlobs`, unlike
`allow.json` / `deny.json`). Schema mirrors the deny overlay's tighten-only
fields: `deny`, `arg_rules`, `secret_patterns`, `block_hosts`, `block_lists`,
`self_modify_globs`. It carries NO `allow` / `allow_prefix` / posture-loosening
field — those are simply absent from the schema, so a widening cannot even be
expressed.

### Admission predicate (the proof)
At launch (and, if a live reload path exists, on reload), the floor assembly:
1. Computes `next = base ∪ self_tighten_overlay`.
2. Computes `delta = diffPolicyWidening(base, next)` — the SAME function the
   reload gate uses (`policy_reload_widen.go:24`).
3. **Admits the self-tighten overlay only when `delta.Empty()` is true.** A
   non-empty widening delta (any added Allow/AllowPrefix, removed Deny, removed
   SelfModifyGlob, or loosened Posture) means the self-write tried to widen — it
   is REFUSED wholesale (not partially applied) and journaled as a rejected swap
   with reason `SELF_WIDEN_REFUSED`.

Because the admission is gated on the widening delta being *empty*, a malicious or
buggy self-write can only ever tighten. The schema shape makes widening
inexpressible; the delta check makes it unenforceable even if the schema is
bypassed. Two independent barriers.

### Journaling
An admitted self-tighten emits a RATCHET-tighten record (sibling to the
`CapabilityGrant` widen record, #5178) with channel `agent-self`, so the exit
summary (#5184) shows "agent tightened its own floor: +N deny, +M arg_rules."

## Mechanism 2 — propose-only widening (#5182)

### Seam
When the agent hits a legitimate `DEFAULT_DENY` (a tool the floor never
enumerated), it may WRITE a structured widening REQUEST to a queue file the
gateway NEVER auto-applies: `.fak/guard/proposals.json`. Each proposal records
the tool name, a why, and the blocked call's context. This file is:
- NOT part of the floor — the floor assembly never reads it as an overlay.
- NOT in `SelfModifyGlobs` — the agent is allowed to write it (writing a request
  grants nothing).

### Ratification (out-of-band only)
`fak manage allow --from-proposals` (sibling to the existing `--from-journal`):
an out-of-band operator reviews the queued proposals and either ratifies (moves a
proposal into the real `allow.json` overlay) or rejects (drops it). Until an
operator ratifies, the proposed tool stays `DEFAULT_DENY`. There is **no
auto-ratification of any kind** — ratification is always a human/operator action
through a channel the agent cannot reach.

### Why this is safe
The proposal file is inert. It is a mailbox, not a floor. The only thing that
turns a proposal into a capability is an operator running `--from-proposals`,
which is the same trust boundary as `fak manage allow` today. The agent gains a
*voice* (structured, ratifiable) without gaining *authority*.

## Invariants (must hold for both #5181 and #5182)

- No self-written file, applied by itself, ever produces a non-empty
  `diffPolicyWidening` against the base floor.
- `self-tighten.json` and `proposals.json` are the ONLY guard files the wrapped
  agent may write; `allow.json`, `deny.json`, the policy manifest, and the embed
  remain in `SelfModifyGlobs`.
- Every admitted self-tighten and every ratified proposal is journaled with the
  `agent-self` channel so the provenance is auditable.
- A red-team test (extending #5174) proves that a `self-tighten.json` attempting
  to add an `allow` entry (via a hand-forged unknown field or a widening
  `self_modify_globs` removal) is refused, not applied.

## Open questions (defer past v1)

- Mid-session live application of a self-tighten (v1 is launch-boundary only,
  matching the TTL work #5179).
- Whether a ratified proposal should carry the originating proposal's context
  into the `CapabilityGrant` provenance (traceability from ask → grant).
