---
title: "Verify, Don't Trust: What DOS Actually Checks"
description: "A DOS primer for newcomers: a done-claim is checked against git, a refusal carries a reason from a closed vocabulary, a recalled memory is re-verified at read time — the three things the kernel re-checks instead of believing."
slug: verify-dont-trust
keywords:
  - verify don't trust
  - DOS
  - trust substrate
  - agent fleet
  - commit audit
  - self-report
  - refusal reason
  - closed vocabulary
  - memory recall
  - verified progress
date: 2026-07-03
---

# Verify, Don't Trust: What DOS Actually Checks

> **TL;DR:** an autonomous agent narrates its own success — it *says* the task
> is done, *says* the refusal was principled, *says* the memory it recalled is
> still true. DOS believes none of it on the agent's word. It re-checks three
> things against evidence the agent did not author: a **done-claim** (against
> git), a **refusal** (against a closed reason vocabulary), and a **recalled
> memory** (against ground truth at read time). The one line to keep: **the
> model proposes, the kernel disposes** — and the kernel checks.

**Concept served:** verify, don't trust (concept 2 of the
[popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md)). This
is the most transferable idea in `fak`: DOS is a domain-free trust substrate for
a fleet of agents, and it works with zero `fak` internals. If you run more than
one agent, you have this problem.

## The problem: a self-narrating worker

Give an agent a task and it will report back. The report is the cheapest thing
it produces and the least trustworthy — it is a *self-report*, authored by the
same process whose work you are trying to grade. A commit subject that says
`fix: handle the null case` is forgeable: the agent wrote both the subject and
the diff, and nothing forces them to agree. "All tests pass" can mean the agent
deleted the failing assertion. "I remembered we already migrated that table" can
be a memory that was true last week and is false now.

In a single-human workflow you catch this by reading the diff. Across a fleet of
agents running concurrently, nobody is reading every diff. DOS is the part of
the system **that does not believe the agents** — it replaces "take the worker's
word" with "check the artifact the worker cannot forge."

![Two columns: left "what the agent says" (a commit subject, "all tests pass", a recalled memory) marked forgeable; right "what git proves" (the diff, the file set, ancestry) marked witnessed; a single blue arrow carries the rule: DOS trusts the right, re-checks the left](../adoption/diagrams/forgeable-vs-witnessed.svg)

*The whole thesis in one image. The left column is what an agent **reports** —
forgeable, because the agent authored the words. The right column is what **git
records** — witnessed, because the agent did not. DOS never passes the left on
its own word: it re-derives the right and checks the claim against it. The three
rows on each side are the three checks named below — a done-claim against the
diff and ancestry, a refusal against a fixed vocabulary, a recalled memory
against the working tree.*

## The three things DOS re-checks

A newcomer should be able to name these three. Each maps to a real `dos` verb
you can run today.

### 1. A "done" claim — checked against git, not the agent's word

When a worker claims a phase shipped, `dos verify PLAN PHASE` answers *did it
actually land* from git ancestry and the run registry — never from the worker's
status message. At the commit grain, `dos commit-audit [REF]` asks the sharper
question: **does this commit's subject match its own diff?** The subject is
forgeable (the agent typed it); the file set the diff touched is not (git
recorded it). So a `fix(auth): …` whose diff only edits a README, or an
`--allow-empty "shipped"`, comes back `CLAIM_UNWITNESSED` instead of passing.

```
dos verify AUTH AUTH2        # did (plan, phase) ship?  exit code IS the verdict
dos commit-audit HEAD        # does the subject match the diff it claims?
```

It grades *did the diff do the kind of thing claimed* — not whether the code is
correct. For correctness you still run the tests; this closes the narrower,
cheaper gap of a claim that does not match its own evidence.

### 2. A refusal — carries a reason from a closed vocabulary

An agent that declines an action should not hand you free-text prose. In DOS a
refusal carries a **reason token from a closed set** — every reason is at once
*emittable* (a worker may stamp it), *verifiable* (an oracle can check the
condition it names), and *refusable* (the loop knows to route it to a replan).
That is what turns "no" into a first-class, auditable value instead of a dead
end. `dos arbitrate` — the admission check that decides whether two workers may
touch the same files — refuses exactly this way, with a structured reason and a
way forward. The vocabulary itself is enumerable, and an unknown token is
treated as drift to fix, not prose to tolerate.

```
dos arbitrate --lane docs --kind cluster   # refuses with a reason token, or admits
```

Because the reason is drawn from a fixed vocabulary, a peer process can *act* on
the refusal — retry a different lane, escalate, replan — without parsing English.

### 3. A recalled memory — re-verified at read time

A saved memory is a frozen self-report from a past session — the least
trustworthy signal in the stack, yet recall hands it to you wearing the
authority of a fact. `dos memory recall NAME` re-checks a memory's concrete
claims (a commit SHA, an import or flag, a file path) against git and the
working tree **now**, at the moment it would be injected — and returns a verdict
(`RECALL_FRESH` / `RECALL_STALE` / `RECALL_UNVERIFIABLE`) rather than the raw
body. A memory that named a file since deleted, or a commit never merged, is
withheld or hedged instead of presented as still true.

```
dos memory recall migrated-orders-table   # is this memory still true right now?
```

## Why it holds: evidence the worker did not author

The through-line under all three: DOS decides from an artifact the worker could
not forge. Git recorded the diff, not the agent. The reason vocabulary is fixed
by the workspace, not by the worker's phrasing. The working tree is ground truth
for a memory, not the memory's own body. Verification is cheap and structural;
trust is expensive and fails silently. DOS spends the cheap thing so a fleet
does not accumulate the expensive one.

## Honest scope

- **This is not a correctness oracle.** `dos commit-audit` checks that a
  commit's claim matches its *diff shape*, not that the code works. Run the
  tests for that.
- **No market-adoption claim is made here.** This page explains a mechanism that
  ships in the repo; it does not assert who uses it. The
  [0/29 prior-art audit](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md)
  frames DOS as a distinctive *assembly* of known primitives, not a novel
  invention.
- **Verbs, not vibes.** Every claim above names a command you can run:
  `dos verify`, `dos commit-audit`, `dos arbitrate`, `dos memory recall`. If a
  claim here doesn't map to a verb, it doesn't belong here.

## Where to go next

- [The tool call is a syscall](tool-call-is-a-syscall.md) — the keystone model
  this sits under: the model proposes, the kernel disposes.
- [Why default-deny beats a classifier](default-deny-vs-classifier.md) — the
  same distrust discipline applied to prompt injection.
- [Memory engineering](memory-engineering.md) — the deeper treatment of
  write-time admission, verified recall, and provable forgetting.
