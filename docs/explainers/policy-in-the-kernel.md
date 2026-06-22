---
title: "The Policy Runs Inside the Kernel"
description: "Most agent safety bolts a recognizer onto the outside of the loop — a hook, a sidecar, an LLM judge — that the model can talk past and that fails open when it breaks. fak puts the permission check on the same code path as the tool call, in one address space, default-deny, with no process to talk past. This is what 'the tool call is a syscall' actually means."
slug: policy-in-the-kernel
keywords:
  - reference monitor
  - capability-based security
  - prompt injection
  - in-process adjudication
  - default deny
  - fail closed
  - LSM
  - agent security
  - tool call
date: 2026-06-19
---

# The Policy Runs Inside the Kernel

> **TL;DR:** `fak` puts the "may this tool run?" check on the same code path as the
> tool call, in one address space, default-deny, with no outside process to crash
> open or argue with. The tool call is a syscall, and the kernel adjudicates it
> before anything happens.

**Short answer:** in almost every agent stack, the thing that decides "may this
tool run?" lives *outside* the loop. It might be a pre-tool hook in another process,
a sidecar policy service over a socket, or an LLM that grades the request. All three
share two weaknesses. The model can argue its way past a recognizer. And when the
outside thing crashes or times out, the call usually runs anyway (fail-open).

`fak` moves that decision onto the *same* code path as the tool call: one Go address
space, no IPC, default-deny. So the check is not a thing the agent talks to. It is a
thing the agent's call *passes through*, the way a `read()` passes through the OS
kernel before it touches a disk. The refusal of an irreversible action does not
depend on catching the attack. It depends on the lever never having been wired up.

That is the whole flip, and it is worth slowing down on. "Policy in the kernel"
sounds like a slogan, but it is actually a specific, checkable claim about *where
the code runs*.

## The thing most systems do: recognize, from outside

Picture the standard shape. The model proposes a tool call. Before it executes,
something inspects it:

- A **pre-tool hook**: a separate program the harness spawns (`exec` a script,
  call out to a gateway).
- A **guardrail / LLM judge**: a second model asked "is this request safe?"
- A **content filter**: a classifier that scores the request or the tool result
  for "looks like an attack."

Every one of these is a *recognizer*. It works by trying to tell good from bad. The
serious prompt-injection research has already reached an uncomfortable conclusion:
recognizing attacks is a losing game. A classifier asks "is this text bad?", and an
attacker with paraphrase, encoding, or a foreign language can make bad text not look
bad. Our own audit of `fak`'s built-in detector measured it as **≈100% evadable** by
a determined attacker, and we say so in the README. A recognizer is a helpful bonus.
It is not a floor.

There is a second, quieter problem that has nothing to do with how smart the
recognizer is: it lives somewhere else. A hook in another process is reached over a
pipe, a sidecar over a socket, a judge over an API. That seam has a default. When
the hook errors, the socket times out, or the judge is slow, what happens to the
call? In most designs, **it proceeds** (fail-open), because failing closed would
wedge the agent on every transient hiccup. So the security property is "we check,
*unless* checking broke," which is exactly when you are under load or under attack.

## The flip: the tool call is a syscall

Here is the reframe `fak` is built on. Treat the model as an untrusted program, the
way an operating system treats application code in ring 3, and treat the harness as
the **kernel**. An untrusted program cannot touch the disk, the network, or another
process's memory directly. It has to make a **syscall**, and the kernel adjudicates
that syscall against permissions the program did not write, before anything happens.

In `fak`, the tool call *is* that syscall. It does not go out to a hook. It goes
through `Kernel.Syscall`, a single in-process chokepoint, where an adjudicator chain
decides Allow / Deny / Defer **before dispatch**: in the same address space, on the
same call stack, with no process boundary in between. The witness that there is no
escape hatch is an *absence* proof. `TestNoOsExecOnHotPath` asserts the decide path
never shells out. There is no other program to be slow, to crash open, or to be
argued with.

This buys three things a recognizer-from-outside cannot have:

1. **There is nothing to talk past.** The model never addresses the gate; its call
   is *subject to* the gate. You cannot sweet-talk a check that isn't a
   conversational participant, for the same reason a process cannot `printf` its way
   into write access to a file it lacks permission for.

2. **The default is closed by construction.** Anything not on the allow-list
   resolves to `DEFAULT_DENY`. An empty policy manifest is the maximally paranoid
   floor: it permits nothing. There is no "the checker was unreachable, so we let it
   through" branch, because there is no remote checker to be unreachable.
   (`TestFoldDefaultDenyEmptyPolicy` pins it.)

3. **The decision is structural, not heuristic.** Whether an irreversible tool runs
   is decided by whether its name is on a reviewable list, rather than by whether a
   model or a regex *recognized* this particular request as dangerous. A list is
   something you can read, diff, and sign. A recall curve is not.

## Why "in-process" is load-bearing, not a micro-optimization

It is tempting to read "in one address space, ~microseconds instead of milliseconds"
as a speed brag. It is not the point, and the project says so: the in-process
adjudication latency is a **subsystem regression sentinel**, not a fleet-speed
headline. For the record, the number is real: roughly a couple of microseconds
in-process versus milliseconds for a spawned-hook baseline on the same box. But
quoting it as "fak is thousands of times faster" would compare against a baseline
nobody actually runs, and would miss what matters.

What matters is that fusing the gate into the loop is what makes the *fail-closed*
default affordable. The reason real systems fail open is that a per-call process
spawn or socket round-trip is expensive and flaky enough that wedging on it would be
worse than the risk. Remove the process boundary and the round-trip, and "refuse if
anything is wrong" stops being a liability. The cheap, local, deterministic check is
what lets default-deny be the *default* instead of an aspiration. Security and the
boundary's cost are the same design knob here, which is the [co-design
thesis](../notes/EXPLAINER-trust-floor-two-lenses-2026-06-17.md) in miniature.

## The adjudicator is a chain, like an LSM — not one filter

"The policy" is not a single `if` statement. It is a ranked chain of small
adjudicators, registered the way the Linux Security Modules framework stacks
security hooks. A new policy rung is `RegisterAdjudicator(rank, impl)`, one more
link in the chain, and the kernel *walks* the registry; it never imports a specific
driver. Each rung can Allow, Deny, or Defer; the chain folds to the most restrictive
verdict, so adding a stricter rung can only ever tighten the floor, never loosen it.
That is why hardening detection is a matter of *composing a driver* (a peer's
normalized-view rung already fronts the base matcher) rather than editing the kernel.

So the picture is not "a filter in front of the model." It is a permission lattice
the call descends through, ranked, fail-closed, every rung in-process.

## Honest scope — what this floor does and does not bound

This is the part to read before citing it, because the flip is powerful exactly to
the degree you are precise about its edges.

- **It bounds tool *names*, structurally.** An irreversible tool you do not
  allow-list is refused regardless of what is in context. That is the guarantee, and
  it holds whether the model is strong, cheap, or actively under attack.

- **It does not, by itself, bound the *arguments* of an allow-listed tool.** If you
  allow a coarse tool like `Bash`, the floor permits `Bash`. It does not, on its own,
  know that `Bash{command: "rm -rf /"}` is the dangerous one. `fak`'s dogfood policy
  *does* ship argument-value deny rules (RE2 patterns that block `rm -rf` while
  allowing `ls`, locked by `TestDogfoodManifestVerdictMatrix`). But those are
  pattern-matching on the command string, which is **detection-shaped**: a determined
  attacker can reword to slip a regex. So the durable advice is to keep irreversible
  tools *off the allow-list* rather than lean on argument matching. Argument-scoped
  *capabilities* (path/host/amount as first-class constraints, rather than regexes)
  are the real fix. They are on the roadmap and not yet shipped.

- **The detector feeding the result-side gate is the evadable part.** The capability
  deny (call-side) and the containment *decision* (result-side quarantine) are
  structural; whether a given poisoned result gets *flagged* is heuristic. `fak`
  makes the decision durable and re-screenable; it does not make the decision smart.

- **The deepest rung is proven but not yet wired live.** The same quarantine verdict can
  evict a poisoned result's K/V span from the kernel-owned attention cache (see
  [addressable KV cache](addressable-kv-cache.md)). But that is proven against a
  synthetic model today. The live `fak agent` path still drives the model over an
  HTTP seam and quarantines at the byte layer. Don't read "policy in the kernel" as
  "the live agent evicts attention state" yet.

The conjunctive bar is the honest summary. An attacker has to beat **two independent
gates**: slip past the evadable screener *and* find an irreversible lever that was
deliberately never wired up. A normal filter is one gate; if it's fooled, you're
compromised. Putting the permission check inside the kernel is what makes the second
gate structural instead of just another recognizer.

## Where to go deeper

- The same mechanism told in two vocabularies (security ↔ optimization), with the
  Rosetta table: [`EXPLAINER-trust-floor-two-lenses-2026-06-17.md`](../notes/EXPLAINER-trust-floor-two-lenses-2026-06-17.md).
- The deployable policy manifest and its exact honest-scope boundary: [`fak/POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md).
- The capability-floor argument-value deny picture: [`SECURITY-capability-floor-2026-06-18.md`](../notes/SECURITY-capability-floor-2026-06-18.md).
- The extension model (how a rung registers without a spine edit): [`fak/ARCHITECTURE.md`](https://github.com/anthony-chaudhary/fak/blob/main/ARCHITECTURE.md).
- Live A/B on real models (injection kept out of context 5/5): `LIVE-RESULTS.md` (private companion).
- The full per-capability honesty ledger: [`fak/CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md).
