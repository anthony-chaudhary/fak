---
title: "fak FAQ — Core concepts and the mental model"
description: "Deep-dive FAQ theme split out of docs/FAQ.md; the essentials and the theme index live there."
---

# Core concepts and the mental model

Part of the [fak FAQ](../FAQ.md) — the essentials and every other theme are
indexed there.

- [llms.txt](https://github.com/anthony-chaudhary/fak/blob/main/llms.txt) — a machine-readable map for LLMs and answer engines.

## Core concepts and the mental model

The ideas the rest of the FAQ builds on: what an agent kernel is, why the model is treated as an untrusted program, and how one boundary carries both security and performance.

## Why does fak treat the language model as an untrusted program?

`fak` treats the model as an untrusted program because its output is shaped by text it reads at runtime — including text an attacker can plant — so nothing the model proposes can count as authorization on its own. The core move puts the model in the position of ring-3 userspace: every effect it wants on the outside world becomes a syscall through a kernel the model does not control, adjudicated from evidence the model did not author, and a tool call is that syscall. The kernel decides allow, deny, transform, or quarantine from a policy floor and the call's own arguments, never from the model's say-so, so an injected instruction can ask for a dangerous action but cannot grant it.

## What does "tool call = syscall" actually mean in fak?

It means every action an agent takes on the outside world is funneled through one in-process checkpoint the model cannot bypass, the way a user-space program reaches the OS only through calls like `read()` or `write()`. In `fak` that checkpoint is the kernel's `Submit`/`Reap` path: a proposed tool call is folded through a ranked adjudicator chain that returns one verdict, and a denied call is never enqueued or executed. Promoting the tool call to a syscall is what lets a single in-process gate mediate both which effects are allowed and which results may enter the model's context.

## What is the "one boundary" idea, and how can the same gate be both security and performance?

The one-boundary idea is that the gate deciding whether a tool result may enter the model's context (a security act) is the same gate that pages that result's bytes to a content-addressed store for reuse (a performance act) — one write-time decision, two enforcement media. When a result is screened, the same code that holds a poisoned result out of context also stores a benign result once in a shared store so shared work isn't recomputed every turn, so the correctness metadata is the performance metadata. `fak` states this as a claim shown by example, not a proven law, and is honest about its edge: the convergence does not help raw GPU throughput (it pays for bit-exactness in memory), and the reuse win only materializes for read-heavy self-hosted fleets.

## If the poison detector is evadable by design, what actually protects me?

The protection is structural — the capability lock and the quarantine policy — not the detector, which `fak` openly calls roughly 100% evadable by design and false-positive-prone. The result screener (`ScreenBytes`, covering secret patterns, injection markers, and byte-repeat pollution) sits on top of the wall as a helpful bonus: if it fires, that's a free catch; if it misses, the result is still held out of context by policy and an unlisted irreversible tool is still refused regardless of context. The honest floor is that the wall holds even when the detector misses, so keep exfil-shaped tools off the allow-list and don't rely on detection as the load-bearing layer.

## What does "in-process" or "in the call path" mean, and why is it load-bearing?

In-process means the permission check runs in the same address space as the agent loop, on the same call path as the tool call, with no spawned hook, no socket round-trip, and no IPC on the decide path. This is what makes fail-closed affordable: there is no per-call process to spawn or socket to wedge on, so the gate can refuse by default without becoming a latency tax you are tempted to turn off. `fak` measures the in-process fold at p50 around 2.4µs versus around 5.8ms for a spawned hook (roughly 2,400×), but it is explicit that this is a subsystem regression sentinel rather than a fleet-speed headline; the point of the number is that the gate is cheap enough to always be on, with absence of process spawn proven by `TestNoOsExecOnHotPath`.

## What is the "trust floor," and why is default-deny the starting point?

The trust floor is the set of effects that are structurally possible at all: a zero or empty policy permits nothing, so every call is refused with `DEFAULT_DENY` until you explicitly allow-list a tool. Default-deny is the starting point because a refusal then does not depend on recognizing an attack — the lever simply was never built, so no context or injection can reach it. You raise the floor deliberately with `allow`, `allow_prefix`, and `deny` rules, and a loaded manifest replaces the floor rather than merging into it; `fak policy --dump` emits the full default to edit and `fak policy --check` validates a manifest before you deploy.

## Does fak stop a tool from being recognized as dangerous, or stop the dangerous thing from existing?

It stops the dangerous thing from existing on the allow-list rather than trying to recognize each attack — the framing is to stop recognizing and start not building the lever. Because an irreversible tool that was never allow-listed has no code path to invoke, an injected instruction can describe the attack perfectly and still get a structural refusal; there is nothing to detect because there is nothing to call. This is why the lock holds against novel phrasings: it is a property of the policy floor, not of a pattern set an attacker can rephrase around.

## What is the honest limit of the capability lock — does it bound tool arguments too?

The lock bounds tool *names* structurally but does not bound the resolved effect of an allow-listed tool's arguments. An allow-listed `send_email` with attacker-chosen recipients, or a coarse `Bash` running `rm -rf /`, is not stopped by the name-level floor — `fak` can inspect one decoded argument string with arg-rules (positive path globs, RE2 deny patterns, byte caps), but RE2 patterns are detection-shaped and evadable, and first-class argument-scoped capabilities (path, host, or amount as constraints) are roadmap, not shipped. The practical guidance is to keep exfil-shaped and irreversible tools off the allow-list entirely rather than trust an argument pattern to catch a bad value.

## How does adding a verdict like "quarantine" fit the same mental model as "deny"?

Both are verdicts in one restrictiveness lattice the kernel folds to, so quarantine (result-side) and deny (call-side) are the same kind of object: a value the next loop turn consumes, not an exception. The adjudicator chain folds to the most-restrictive verdict across allow, defer, transform, quarantine, require-witness, and deny; an unknown verdict kind fails closed rather than panicking, and a refusal is returned as a structured result, never an HTTP error. That uniformity is why a result quarantine and a call denial share one wire shape and one audit path: the model proposed something, the kernel returned a verdict, and the loop reads it in-band.

## What is memory engineering?

