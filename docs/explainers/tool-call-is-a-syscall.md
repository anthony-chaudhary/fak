---
title: "The Tool Call Is a Syscall"
description: "The one-page mental model behind fak: an OS kernel never trusts a user program's word that a write is safe — the syscall crosses a boundary the program does not control. fak does the same to the LLM. The model proposes a tool call; the kernel adjudicates it before anything happens. Model proposes, kernel disposes."
slug: tool-call-is-a-syscall
keywords:
  - tool call
  - syscall
  - agent security
  - default deny
  - kernel boundary
  - adjudication
  - prompt injection
  - quarantine
  - model proposes kernel disposes
date: 2026-07-02
---

# The Tool Call Is a Syscall

> **TL;DR:** an operating system never asks a program to promise it will behave —
> every dangerous action must cross into the kernel, where the program has no vote.
> `fak` puts the same boundary under an AI agent: every tool call is checked by code
> the model cannot talk to, talk past, or turn off. **The model proposes; the kernel
> disposes.**

## Start with the computer you already trust

Your laptop runs programs you have never read, written by people you have never met.
That is fine, and the reason it is fine is fifty years old.

A program can compute whatever it likes in its own memory. But the moment it wants to
touch the *world* — write a file, open a network connection, kill a process — it
cannot just do it. It has to ask. That ask is called a **syscall** (system call): the
program stops, control crosses into the **kernel** — the operating system's core,
which the program does not control — and the kernel decides whether this program, with
these permissions, may do this thing. The program's opinion of its own request carries
no weight. A program that *says* it is deleting a temp file but *asks* to erase your
home directory gets judged on the ask, not the story.

That one design choice is why you can run untrusted code at all. Safety does not come
from reading the program's mind. It comes from a boundary the program cannot cross by
being persuasive.

## Now replace "program" with "model"

A large language model is a text engine: on its own it can only *say* things. The
moment it is given tools — run a shell command, edit a file, call an API, send money —
it can *do* things, and it becomes exactly what a program is to an operating system:
useful, capable, and not to be taken at its word. It can be wrong. It can be tricked
by text it read on the way (a **prompt injection** — instructions smuggled into a web
page or file the model was merely supposed to summarize). Its stated intentions are
part of its output, and its output is the thing in question.

Most agent stacks still execute the model's tool calls on trust, or guard them with
another model asked "does this look safe?" — a judge that can itself be argued with.

`fak` applies the fifty-year-old answer instead. A **tool call** — the structured
"run this tool with these arguments" message the model emits — is treated as a
syscall: it must cross a boundary into a small kernel the model does not control,
and nothing happens until that kernel rules on it.

## The path of one call: proposed → verdict → (maybe) executed

`fak` is one Go binary that sits between the agent and the world (`fak guard --
claude` puts it there in one command). Every tool call walks this path:

1. **The model proposes.** It emits a tool call — a request, not an action.
2. **The call crosses the boundary.** It lands in the kernel's adjudicator, in-process
   — no second service, no network hop. The measured cost is **~362 ns per decision**
   (Apple M3 Pro): the check is free at the timescale of any real tool.
3. **The kernel rules against a capability floor.** The floor is a reviewable
   allow-list that is **default-deny**: a tool or action not explicitly granted cannot
   run — no matter what the model says, or what an attacker convinced it to say. The
   verdict is one of a closed set — **allowed**, **denied**, **repaired** (rewritten
   to a safe form), or **deferred** (held for a human). Every refusal carries a
   machine-checkable reason code (`DEFAULT_DENY`, `POLICY_BLOCK`, `SECRET_EXFIL`, …),
   not prose.
4. **Only an allowed call executes.** And what comes *back* is judged too:
   a suspicious tool result can be **quarantined** — held out of the model's context —
   so poisoned output never becomes the model's next instructions.

At exit you get the ledger, one line:
`fak guard: 131 kernel decisions; 121 allowed / 5 denied / 2 repaired / 0 quarantined / 3 deferred`.

![Left-to-right flow: the agent emits a tool call; the fak kernel adjudicates it against a default-deny capability floor; four verdicts branch out — ALLOW (the call runs), DENY (it never reaches the tool), TRANSFORM (rewritten to a safe form, then runs), REQUIRE_WITNESS (held) — and any result that runs is checked again before re-entering context](../adoption/diagrams/syscall-flow.svg)

*The whole path on one frame. The model only **proposes**; the kernel **disposes** —
a call the floor denies never reaches the tool runner (call-side gate), and a result
the floor distrusts is **quarantined** before it can become the model's next
instructions (result-side gate). The four verdicts on the right are exactly the four
numbers in the ledger line above: allowed / denied / repaired / deferred.*

To see the flow in motion, watch the
[44-second agent-kernel explainer](../../visuals/agent-kernel-video.mp4).

## Why the placement is the whole point

Notice what this does *not* rely on: recognizing attacks. `fak`'s own injection
detector is measured **≈100% evadable** by a determined attacker, and the docs say so
— every detector is. The floor holds anyway, because refusing a dangerous action does
not require *catching* the trick; the lever was never wired up. An OS kernel does not
detect malware to stop a permission violation, and neither does this. The deeper
engineering story — why an in-process, fail-closed check beats an external hook or an
LLM judge that fails open — is in
[Policy in the kernel](policy-in-the-kernel.md). None of this is a new invention;
it is the oldest idea in operating systems, finally applied to the newest kind of
program.

## Try it in ten seconds

No key, no model, no GPU — ask the kernel directly:

```bash
fak preflight --tool refund_payment --args "{}"
# -> DENY (DEFAULT_DENY): not on the allow-list, fail-closed
```

## The sentence to keep

**The model proposes; the kernel disposes.** If you remember one thing, remember that
— and when someone asks how an AI agent can be trusted with real tools, answer the way
an operating system would: it isn't trusted; it's adjudicated.

---

*Next:* [Policy in the kernel](policy-in-the-kernel.md) — the engineering depth ·
[Engineering is building loops](engineering-is-building-loops.md) — the loop this
syscall sits inside · [README](../../README.md) — the whole project in one read.
