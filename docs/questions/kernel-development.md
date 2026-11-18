---
title: "Kernel development — the questions a newcomer actually asks"
description: "The question ledger for the kernel-development axis: the questions a newcomer asks when they first try to change fak's kernel, each with a short answer and a pointer to the rung of the kernel-development ladder that answers it in full."
slug: questions-kernel-development
keywords:
  - kernel development questions
  - frozen ABI
  - registration seam
  - correctness class
  - non-forgeable keep-bit
  - adjudicator rung
  - question loop
  - kernel development ladder
  - agent kernel vs CUDA kernel
date: 2026-11-18
---

# Kernel development — the questions a newcomer actually asks

> **What this is.** The question ledger for the **kernel-development** axis. Not a
> tutorial — a list of the questions a newcomer actually asks the first time they try to
> change fak's kernel, each with a short answer and a pointer to the rung of the
> [kernel-development ladder](../explainers/kernel-development/README.md) that answers it
> in full. The page *is* the questions; the answers are deliberately short, because the
> rung is where the real answer lives.

"Kernel development" here means one specific practice: **changing the code the kernel
walks, without forking the spine.** A newcomer meets that sentence and immediately has
questions — some contrarian ("isn't a frozen ABI just bureaucracy?"), some afraid ("what
if I break the spine for everyone?"), and some nobody has thought to ask ("who decides
what *correct* means?"). This page collects them, labels them honestly, and routes each
one to the rung that closes it.

## How this page works

This page is the durable surface of the **question loop** — a super-loop-family member
whose whole job is to *ask* hard questions and *not* answer them. Two loops, in two
separate context windows, keep it honest:

- **The ASK loop** (role: *questioner*) writes the questions. Each tick it appends a
  handful of `open` rows to [`asked.jsonl`](asked.jsonl) and stops. It never proposes
  fixes, files tickets, or answers itself.
- **The NEXT-STEP loop** (role: *ticketer*) turns an unanswered question into work. Each
  tick it takes **one** `open` question, decides its fate, and — only if the question
  earns it — files **one** ticket.

**Why the two loops are separated.** A questioner that also holds the ticketing pen
slides into solution mode: it starts writing questions it already knows how to close, and
quietly stops asking the uncomfortable ones. The load-bearing rule is that **the
questioner must not be the one who decides the question is answered.** Reading the open
backlog is fine and necessary; *triaging* it is what corrupts the ASK role. A fresh
context per ASK tick keeps the provocations honest. The full schema, the four categories,
and the machine-checked ledger rules live in the [question-loop README](README.md).

The categories below are the ledger's own vocabulary — **CONTRARIAN**, **AFRAID**,
**UNASKED**, and **STEELMAN** — not a new taxonomy invented for this page.

## CONTRARIAN — questions that challenge what the repo claims is correct

### Isn't a frozen ABI just bureaucracy?

A frozen ABI looks like red tape until you count the importers. `internal/abi` is the one
tree *every* leaf and *every* command imports, so an edit to it collides every in-flight
change and forces a fleet-wide recompile. The freeze is not a preference for stability —
it is the property that lets a hundred workers build at once without stepping on each
other. The bureaucracy is the point: it is what makes parallel work possible.

→ full answer: [Level 2 — The registration seam](../explainers/kernel-development/level-2-the-registration-seam.md)

### Why can't I just add a field to the struct?

Because the struct is the shared vocabulary, and a field added for your feature is a field
every other worker now has to compile against. The repo's answer is the **additive
envelope**: a new field rides a `Capability`-guarded extension slot, so old workers that
never negotiate the capability are unaffected. You *can* extend the wire — you just cannot
extend it in a way that breaks a peer who has not heard of your idea yet.

→ full answer: [Level 2 — The registration seam](../explainers/kernel-development/level-2-the-registration-seam.md)

### Isn't the keep-bit just a benchmark I could game?

The keep-bit is deliberately **non-forgeable**: it is unexported and set only from a
measured witness, so you cannot write it from your own code. But the honest answer is
narrower than "ungameable" — a passing keep-bit proves a *strict measured gain on one
host, in one window, against one baseline*. It does not prove the win generalizes. The
gate removes the author's opinion from the decision; it does not remove the author's
choice of what to measure.

→ full answer: [Level 4 — Prove faster](../explainers/kernel-development/level-4-prove-faster.md)

### Isn't the correctness class just a label I pick to pass the looser gate?

It is a label — but a *typed* one, and the harness reads it mechanically. `Reference` and
`Approx` are real Go values, and `compute.RequireReference` is called *before* any
bit-identity assertion, so it is impossible to hold a device to a gate it cannot meet or
to quietly promote a device to `Reference`. You declare your class honestly; the right
gate then applies itself. Picking the looser class is not cheating — it is the correct
declaration for any kernel that reorders floats or quantizes a weight.

→ full answer: [Level 3 — Prove correct](../explainers/kernel-development/level-3-prove-correct.md)

## AFRAID — the elephant in the room

### What if my kernel is correct but slower?

That is the exact failure mode Gate 3 exists to catch, and it is the reason a
correct-but-slower kernel cannot land. It passes every correctness witness, it *looks*
like progress, and it quietly makes the system worse — so the keep-bit refuses it. The
fear is well-founded and the answer is structural: you do not get to land a regression
just because it is correct.

→ full answer: [Level 4 — Prove faster](../explainers/kernel-development/level-4-prove-faster.md)

### What if I break the spine for everyone?

You cannot, and that is by design. The spine is frozen and no worker may lease it; a new
idea attaches as a new package plus one `Register*()` call, never an edit to the core. The
`internal/architest` layering gate fails the build on an upward or cross-tier import, so a
change that would reach into the spine is refused before it lands. The fear is the reason
the freeze exists.

→ full answer: [Level 2 — The registration seam](../explainers/kernel-development/level-2-the-registration-seam.md)

### What if my numbers don't reproduce on the reviewer's machine?

Then your keep-bit was an observation, not a claim. A reproducible win needs a *named
honest baseline* (tuned/warm, same host and session), an *identical measurement window*
for both arms, the measured resource freed first, and a clean truth so the result
re-derives from committed state. The naive/cold and tuned/warm baselines are never
blended, because averaging two different questions yields a number true of nothing.

→ full answer: [Level 4 — Prove faster](../explainers/kernel-development/level-4-prove-faster.md)

## UNASKED — the blind spot nobody has raised

### Who decides what "correct" means?

Not you, and not a reviewer — a *declared class* plus a deterministic witness. You write
the kernel and the witness; the toolchain re-runs the witness on every change, and a
kernel whose witness goes red cannot land. "Correct" is not an opinion anyone holds; it is
the property a re-runnable command asserts. The authority is the witness, and the witness
is authored by the harness, not the author.

→ full answer: [Level 3 — Prove correct](../explainers/kernel-development/level-3-prove-correct.md)

### What happens to my kernel when the ABI version bumps?

Nothing, if you attached through a seam. The seam is frozen; the backend behind it may
move. A new verdict kind registers for `k > 1023` with no core edit, a new field rides a
capability-guarded envelope slot, and a new backend is a new file plus one `init()`. The
rule of thumb from the frontier rung: **the seam is frozen, the backend behind it may
not be** — so build against the seam, not against today's implementation.

→ full answer: [Level 6 — The frontier](../explainers/kernel-development/level-6-the-kernel-development-frontier.md)

### Is the agent kernel the same thing as a CUDA kernel?

No — the word "kernel" is doing three jobs in this repo. The **OS-metaphor kernel** is the
reference monitor that adjudicates tool calls, in pure Go on the CPU. The **compute
kernel** is the inner-loop math a matmul dispatches to. The **CUDA kernel** is a literal
`__global__` function launched onto the GPU. The three never overlap: no GPU code
participates in a tool-call verdict, and the reference monitor never touches the forward
pass.

→ full answer: [Level 1 — What is kernel development?](../explainers/kernel-development/level-1-what-is-kernel-development.md)

## STEELMAN — the strongest form of the opposite side

### Steelman: a monorepo with no frozen spine would let me ship faster — what does the freeze actually buy?

The steelman is real: with no freeze you could add a `case` to a switch and ship in an
afternoon, with no registration ceremony. What the freeze buys is *parallelism* — because
each leaf is its own directory and the spine never moves, two workers can build two ideas
at once and edit disjoint files. The freeze trades a little per-change speed for the
ability to run a fleet of changes at once without collisions. If you are one person
shipping one change, the steelman wins; if you are a fleet, the freeze does.

→ full answer: [Level 2 — The registration seam](../explainers/kernel-development/level-2-the-registration-seam.md)

### Steelman: a human reviewer's judgment beats a mechanical keep-bit — why trust a gate over a person?

The steelman is that a reviewer can weigh context a gate cannot: is this win worth its
complexity, does it generalize, is the baseline honest? The repo's counter is that the
author is the worst verifier of their own speedup, and a reviewer reading a diff is
reading the author's framing. The gate does not replace judgment — it removes the
author's *opinion* from the KEEP/REVERT decision, and leaves the judgment questions
(worth, generality) explicitly outside what a passing keep-bit proves.

→ full answer: [Level 4 — Prove faster](../explainers/kernel-development/level-4-prove-faster.md)

### Steelman: the admission check should be a separate process, not in-kernel — what does in-process actually buy?

The steelman is isolation: a separate process cannot be bypassed by a bug in the caller,
and a crash in the check cannot take down the agent. The repo's counter is that the
reference monitor must be the code a user program *cannot* bypass — and an out-of-process
check is a check the caller can route around. In-process is what makes the fold a
chokepoint: the model proposes, the kernel disposes, and a `DENY` never reaches dispatch.
The cost is that the check shares the agent's address space; the benefit is that it cannot
be skipped.

→ full answer: [Level 5 — The adjudicator rung](../explainers/kernel-development/level-5-the-adjudicator-rung.md)

## Questions still open

These are questions the repo genuinely cannot answer yet. They are marked **open** in the
ledger, and each names what would close it.

- **Is the keep-bit a valid gate for an adjudicator rung, or only for a compute kernel?**
  The keep-bit machinery (`shipgate.Evaluate`) is compute-shaped today: it measures a
  throughput win, while a rung's cost is a per-call latency. *What would close it:* a
  rung-shaped keep-bit with a measured per-call witness, and a worked example of a rung
  clearing it.
- **Does the "add a rung" path have the same mechanical witness per gate as the "add a
  backend" path?** Rungs 1–4 have a mechanical witness per gate; rung 5's rungs are
  witnessed too, but the *add-a-rung* path is younger than the *add-a-backend* path.
  *What would close it:* an end-to-end, witnessed example of adding an adjudicator rung
  from scratch.
- **Can a real external engine's KV region actually be attached through the frozen
  `RegisterRegionBackend` ABI?** The ABI is frozen; the transport behind it is a `[STUB]`.
  *What would close it:* building the CUDA-IPC / shared-memory import behind the frozen
  seam and witnessing it against a live engine.
- **Is the fine-tuned adjudication model real, or a stock forward pass wearing the name?**
  The fused model is a *stock* reference forward pass; a tuned head is unbuilt and tagged
  `[STUB]`. *What would close it:* training or grafting a tuned head onto the fused model
  and witnessing the verdict quality against the stock baseline.

## See also

- [The question-loop ledger](README.md) — the schema, the four categories, and the
  machine-checked rules behind this page.
- [The kernel-development ladder](../explainers/kernel-development/README.md) — the six
  rungs every "→ full answer" above points into.
- [What is a CUDA kernel?](../explainers/what-is-a-cuda-kernel.md) — the three-senses
  disambiguation the "is the agent kernel a CUDA kernel?" question rests on.
- [`EXTENDING.md`](../../EXTENDING.md) — the task-first golden path: plug in → prove
  correct → prove faster → ship.

---

*[← back to the question-loop README](README.md) · [the kernel-development ladder](../explainers/kernel-development/README.md)*
