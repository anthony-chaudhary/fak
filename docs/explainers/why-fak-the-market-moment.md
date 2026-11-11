---
title: "Why fak, now: the market moved from writing code to checking it"
description: "Agent iteration collapsed from days to minutes, so the bottleneck migrated downstream: review throughput, verification, and trust now cap what agent fleets ship. The market offers two binaries - gate every step by hand, or just trust the model - and both fail at fleet scale. fak is the third option: a runtime where you define what you trust and the kernel proves the rest."
slug: why-fak-the-market-moment
keywords:
  - agentic development
  - PR review bottleneck
  - trust boundary
  - default deny
  - graduated autonomy
  - agent fleets
  - verification
  - governance
  - local inference
date: 2026-11-10
---

# Why fak, now: the market moved from writing code to checking it

> **TL;DR:** the coding-agent market crossed from "can it write code?" (settled -
> yes) to "what checks the result?" (open). Faster iteration cycles moved the
> bottleneck downstream: agents can produce changes faster than humans can review
> them, so PR review, verification, and trust - not typing - now cap shipped work.
> The market's two live answers are binary and both fail at fleet scale: gate
> every step by hand and the human becomes the serialization point, or just trust
> the model and lose the safety net that made the speed usable. fak is the third
> option: a runtime where trust is a *definition you write down* - a default-deny
> tool floor, done-claims checked against git evidence, kernel-arbitrated leases
> for parallel agents - instead of a leap of faith or a human in every hot path.

**Concept served:** the motivation frame for the whole runtime - why the tool
floor, the verdict kernel, evidence-checked done-claims, and fleet serving exist
at all. This page is the *why*; the *how* lives in the sibling explainers linked
below.

**The pieces.** This is a modular narrative: seven self-contained pieces, each of
which stands on its own. The rendered presentation is
[`market-moment.html`](../market-moment.html).

| # | Piece | The one line |
|---|---|---|
| 01 | [Where the market is today](#piece-01) | agents are a daily driver; the unit of work is shifting to machine-paced change with human checkpoints. |
| 02 | [The bottleneck migrates downstream](#piece-02) | generation compresses to minutes; review, verification, and trust stay human-paced - the queue moves. |
| 03 | [The two answers, and why both fail](#piece-03) | gate everything by hand (serial) or trust the AI (forgeable); audit observes without governing. |
| 04 | [The middle ground fak builds](#piece-04) | trust as a definition, not a leap: six binary-to-middle paths, each a mechanism you can run. |
| 05 | [Born from real problems](#piece-05) | each mechanism is the scar of a specific observed failure, not a forecast. |
| 06 | [The serving side](#piece-06) | fleets change what inference must do: batch, reuse, fail over, preempt, meter. |
| 07 | [What this frame is not](#piece-07) | not "fully autonomous," not "review is dead," not "every layer is finished." |

---

<a id="piece-01"></a>
## 01 · Where the market is today

Coding agents are no longer a demo layer. They are a daily driver: harnesses run
inside terminals and IDEs, file real PRs, and increasingly run unattended on
recurring repo work. The industry conversation has moved with them - from
benchmarking models on code generation to governing what agents are allowed to
do (the existence of an OWASP Agentic Top-10 and an MCP Top-10 is the symptom:
a risk list is published only for a practice that is already widespread).

The unit of software work is shifting with it. The old unit was "a developer
writes a change and files a PR, peers review it." The new unit is "agents
produce most of the change; humans direct, review, and own the decisions that
matter." Nobody has to believe the end state for the trend to matter - only
that the *direction* is machine-paced change with human checkpoints.

<a id="piece-02"></a>
## 02 · Where it is moving: the bottleneck migrates downstream

Walk the loop and watch where the time goes.

**The old loop:** think → write → test → PR → review. Days per cycle. The human
wrote the code, so review load scaled with writing load - one human, one
reviewer, roughly matched.

**The agent loop:** the generation side of that cycle compresses to minutes. An
agent drafts, tests, and re-works a change while you read its first summary. A
fleet of agents runs many such cycles in parallel lanes at once.

Nothing in that compression makes the *human checkpoints* faster. Review
throughput, acceptance testing, and trust decisions stay human-paced while
generation becomes machine-paced. The queue does not disappear - it moves. The
faster the iteration cycle, the more PRs pile up at the review gate, and the
more the human becomes the serialization point in an otherwise parallel system.
A fleet that can produce fifty changes a day behind a review gate that absorbs
five is a five-change-a-day fleet. The speedup is real and it is thrown away at
exactly one step.

So the market's open question is no longer whether agents can write code. It is
**what is the review-and-trust substrate for machine-paced change** - and the
current answers do not fill that gap.

<a id="piece-03"></a>
## 03 · The two answers the market offers today, and why both fail at fleet scale

**Answer 1: gate everything by hand.** A human reviews every PR, approves every
tool call, verifies every "done." This is the world most teams live in, and it
rebuilds the old bottleneck one step downstream: review load now scales with
agent output instead of human output. It is safe, it does not scale, and it
cancels the very speed the agent was bought for.

**Answer 2: just trust the AI.** Auto-accept the tool calls, auto-merge the
PRs, read the agent's summary as ground truth. This removes the human queue -
and removes the safety net at the same time. The failure modes are the ones the
agent-governance risk lists exist for: a silent breakage nobody read, a "tests
pass" that was never run, an instruction smuggled in through fetched content.
And it fails in a way that is worse than slow: *unauditable*. When the agent
grades its own homework, neither the human nor the next agent can tell which
parts of the story were checked.

**The bolt-on middle that isn't one:** after-the-fact audit and observability
platforms. They record what the agent did and alert afterward - useful, but
they *observe* the loop without *governing* it. A log entry does not stop the
bad tool call; it files the report. Prevention and audit are different layers,
and the market mostly sells the second.

What is missing is the layer between: a runtime that governs the boundary
*while the agent runs* - fast enough not to become the new human queue, and
defined enough not to be a leap of faith.

<a id="piece-04"></a>
## 04 · The middle ground fak builds: trust as a definition, not a leap

The binary framing - "a human has to do it" or "just trust the AI" - treats
trust as a switch. It is actually a dial, and the design space between the
extremes is where fleets become usable. fak is machinery for making that dial
precise. Each mechanism moves one class of decision off both extremes and onto
the kernel:

| The binary | The middle path fak implements |
|---|---|
| Human runs every tool ↔ model calls anything | **A capability floor on every tool call.** Structural denies and danger rules refuse by structure under every posture, and the strict `fail_closed` posture lets you enumerate exactly what the agent may do — everything else refused, without a recognizer that can fail open. ([why default-deny beats a classifier](default-deny-vs-classifier.md)) |
| Human verifies every done-claim ↔ believe the agent's summary | **The kernel re-checks the claim against evidence.** A done-claim is graded against git - the subject is forgeable, the diff is not. ([verify, don't trust](verify-dont-trust.md)) |
| Human coordinates who touches what ↔ hope parallel agents don't collide | **Lane leases and arbitration.** The kernel decides whether two agents may run at once on the same files, and refuses with a structured reason a peer can act on. ([lane leases](../agentic-issue-dispatch.md), [lane arbiter](../cli/sync.md)) |
| Human approves every exception forever ↔ silent override | **Graduated exceptions.** An attending operator can approve *one exact refused call* - the agent cannot approve itself ([operator approval](../notes/GUARD-OPERATOR-APPROVAL-2026-10-23.md)); blocked gates have recorded, expiring, scope-bound waiver paths instead of a permanent lock or a silent bypass. |
| Human triages every failure ↔ agent retries blindly | **Structured refusals from a closed vocabulary** - a machine-checkable reason token, not free-text "I couldn't do it." ([verify, don't trust](verify-dont-trust.md)) |
| Trust the model's word about what ran ↔ read raw logs yourself | **The verdict fold at the boundary.** Proposed call → verdict → only an allowed call executes; suspicious results are quarantined. ([the tool call is a syscall](tool-call-is-a-syscall.md)) |

The pattern in every row: the human keeps the *decision* (what is allowed, what
a gate means, whether this exception is worth its risk) and stops being the
*serialization point* (approving each instance, coordinating each pair of
agents, reading each log). Policy moves into the kernel; judgment stays with
the person. That is the middle ground - not "trust the AI," not "do it as a
human," but **define what you trust, and make the runtime prove the rest.**

The sentence to keep: **the model proposes, the kernel disposes - and the
kernel checks.**

<a id="piece-05"></a>
## 05 · Born from real problems, not from a thesis

None of the mechanisms above were designed forward from a vision of autonomous
agents. Each is the scar of a specific failure observed running agent work for
real, day after day:

| The failure that actually happened | The mechanism it forced |
|---|---|
| Two agents edited the same files at once and trampled each other's work | Lane leases + arbitration before a worker takes a lane |
| An agent reported "done" and the work was not done | Done-claims verified against git evidence, not the worker's word |
| A worker sat blocked behind a gate nobody would ever clear | Time-boxed, scope-bound exception paths instead of indefinite human locks |
| A refused action came back as prose nobody could route on | Refusals as tokens from a closed reason vocabulary |
| Injected content tried to drive a tool call | Default-deny floor + result quarantine at the boundary |
| Long sessions drowned in their own accumulated context | Context management that sheds stale history instead of hoarding it ([context](cache.md)) |
| One paused model provider stopped every worker at once | Router failover across providers ([provider failover audit](../gateway-governance-parity-audit.md)) |
| A serial server turned a multi-turn agent into a waiting queue | Batching, prefix reuse, and capacity admission with backpressure ([continuous batching](../supported/features.md), [KV cache for agentic context](kv-cache-agentic-context.md)) |

This is the honest claim, and it is a stronger one than a prediction: **fak is
not a forecast about what agents will need - it is the recorded list of what
broke when agents ran for real.** The roadmap is downstream of the scars, not
ahead of them.

<a id="piece-06"></a>
## 06 · The serving side of the same story

Fleets change what inference has to do, not just how much of it. A fleet's
serving layer must batch concurrent lanes into one decode, reuse the prefixes
that multi-turn tool loops share, fail over when a provider pauses, let the
operator's interactive turn preempt background batch work, and meter against
budgets - or the coordination layer above it starves. fak's serving runtime
exists because those are the observed failure modes of agent serving, not
speculative ones; see the [capability floor](../../docs/supported/features.md)
and the [honest boundary](what-fak-is-not.md) for what is measured today versus
what is not. Running the loop on your own machine keeps it fast and keeps the
code local - the same trust argument applied to data.

<a id="piece-07"></a>
## 07 · What this frame is not

- **Not "fully autonomous."** The durable guarantee is the enforcement layer,
  not the model's judgment; the human keeps review responsibility for the
  decisions that matter. fak moves work off the human's desk; it does not
  remove the human.
- **Not "review is dead."** Review changes shape: from reading every change to
  defining what the runtime must prove about every change, then sampling the
  residue with evidence in hand.
- **Not a claim that every layer is finished.** What is measured, limited, or
  planned is tracked in [CLAIMS](../../CLAIMS.md) and the
  [status page](../../STATUS.md); the motivation frame does not upgrade any
  claim tier.

## Where to go next

- The keystone mental model: [the tool call is a syscall](tool-call-is-a-syscall.md)
- What the kernel actually checks: [verify, don't trust](verify-dont-trust.md)
- The load-bearing guarantee: [why default-deny beats a classifier](default-deny-vs-classifier.md)
- The volunteered limits: [what fak is not](what-fak-is-not.md)
- Try it: [the README](../../README.md) - `fak up` for a local runtime, `fak guard` to wrap an agent you already run.
