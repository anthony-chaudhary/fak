---
title: "The fak/DOS Glossary: 12 Terms in One Line Each"
description: "Plain one-line definitions of the vocabulary a newcomer trips on — verdict, capability floor, quarantine, addressable KV cache, bit-exact eviction, witness, lease, refusal reason, recall re-verification, syscall boundary, prefix-cache discount, fail-closed — each linked to its full explainer."
slug: glossary
keywords:
  - fak glossary
  - DOS glossary
  - agent kernel vocabulary
  - verdict
  - capability floor
  - quarantine
  - addressable KV cache
  - bit-exact eviction
  - witness
  - refusal reason
  - fail-closed
  - syscall boundary
date: 2026-07-03
---

# The fak/DOS glossary — 12 terms, one line each

The vocabulary a newcomer trips on, defined plainly. Each term links to the
explainer that earns the definition. People can only repeat what they can
name — this page is the naming.

## The boundary

- **[Syscall boundary](tool-call-is-a-syscall.md)** — the line a tool call crosses out of the model's hands: like a user program calling the OS, the agent proposes an action and a kernel it does not control decides — the model proposes, the kernel disposes.
- **[Verdict](tool-call-is-a-syscall.md)** — the kernel's ruling on one proposed tool call (allow, refuse, quarantine), produced by an inspectable check — never by asking the model whether its own call looks safe.
- **[Capability floor](default-deny-vs-classifier.md)** — the default-deny list of what an agent may do at all: anything not explicitly granted is refused by structure, so an attack is stopped without having to be recognized.
- **[Quarantine](default-deny-vs-classifier.md)** — where a suspicious tool result goes instead of straight into the model's context: held apart, screened, and admitted only if clean.
- **[Fail-closed](policy-in-the-kernel.md)** — when a check cannot run or cannot decide, the answer is "no": an error means refusal, never a silent pass-through.

## The trust substrate (DOS)

- **[Witness](verify-dont-trust.md)** — evidence the system recorded (a git diff, a journal row) rather than a claim an agent narrated; a "done" counts only when a witness backs it.
- **[Refusal reason](verify-dont-trust.md)** — a "no" carrying a token from a closed vocabulary (e.g. `LANE_DRAINED`) instead of free prose, so a peer can verify the condition and act on it.
- **[Lease / arbitration](verify-dont-trust.md)** — how parallel agents avoid collisions: each declares the file region it will touch and is admitted only if that region is disjoint from every live lease.
- **[Recall re-verification](memory-engineering.md)** — a saved memory is re-checked against git and the working tree at the moment it is read; a stale claim is withheld or hedged, never injected as fact.

## The cache

- **[Addressable KV cache](addressable-kv-cache-in-5-min.md)** — a context cache you can reach into the middle of, not just append to: one span can be removed after the fact instead of rebuilding everything behind it.
- **[Bit-exact eviction](addressable-kv-cache.md)** — removing a span leaves the cache bit-for-bit identical to one that never contained it (`max|Δ| = 0`): provable forgetting, not best-effort deletion.
- **[Prefix-cache discount](long-session-economics.md)** — the provider's cheaper rate for a prompt prefix that is byte-identical to what it has already processed; it survives only while those bytes do not change.

## Related glossaries

This page is the newcomer's on-ramp. Two sibling glossaries go deeper on
different axes: [the canonical split of overloaded core words](../glossary.md)
(session, agent, context, model, memory) and
[the managed-context glossary](../managed-context-glossary.md) (the product
contract for what fak manages automatically).
