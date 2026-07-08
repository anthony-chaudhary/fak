---
title: "The Status a Peer Can Trust: Verified Progress for In-Flight Agents"
description: "Every A2A protocol standardizes how one agent reports its status to another — and every one of them ships that status as a self-report the reader is asked to believe. fak ships the opposite: a peer-readable run digest with no `claimed` field by construction, so an in-flight agent hands a successor only progress it can re-verify against git and the intent ledger."
slug: status-a-peer-can-trust
keywords:
  - agent-to-agent
  - A2A protocol
  - in-flight agents
  - self-report
  - verified progress
  - no claimed field
  - execution provenance
  - agent status
  - dos_status
  - Agent2Agent
date: 2026-07-07
---

# The Status a Peer Can Trust: Verified Progress for In-Flight Agents

> **TL;DR:** two agents running at the same time have to tell each other *what
> is happening right now* — am I still moving, how far did I get, is it safe for
> you to start. Every A2A protocol standardizes the **shape** of that message and
> leaves its **truth** to the sender: the status is a self-report, authored by
> the same process whose progress you are trying to grade. fak's answer is a
> run digest with **no `claimed` field by construction** — a peer is handed a
> pointer to progress the kernel *verified* (a git commit, an intent-ledger row),
> never a number the worker asserted. The one line to keep: **status is a
> re-verifiable cursor, not a self-report.**

**Concept served:** this is the in-flight, agent-to-agent sibling of
[Verify, don't trust](verify-dont-trust.md). That page is about grading a worker
*after* it stops (did the commit land, is the memory still true). This one is
about the harder, live case: a *running* agent B needs to read a *running* agent
A's state and act on it — before either has finished, with no human reading the
diff in between.

## The problem: an in-flight worker narrates its own status

Give an agent a long task and set a second agent to depend on it. The dependent
agent needs to know: is the first one still alive, how far has it got, is the
file region it holds safe for me to touch yet. The cheapest way to answer is to
ask the first agent — and its answer is the least trustworthy signal in the
system, because it is a *self-report* produced by the process whose progress you
are trying to grade. "I'm 80% done" can be a hopeful guess. "Working" can mean
wedged in a retry loop. "Completed" can mean it deleted the failing assertion.
The **confidence–accuracy gap** — agents reporting high confidence in wrong
answers — is by now a documented failure mode of multi-agent systems, not an
edge case ([survey](https://arxiv.org/abs/2606.04990)).

In a single-human workflow you catch this by reading over the worker's shoulder.
Across a fleet of agents running concurrently, nobody is watching every worker.
So the question is not "how do we transport a status message" — that is the easy
half, and the whole industry has solved it. The question is: **when agent B reads
agent A's status, can B believe it without trusting A?**

## What the standards actually standardize: the self-report

The 2025–2026 wave of agent-interop protocols all answer the *transport* half and
punt on the *trust* half:

- **Google's A2A** (Agent2Agent, now under the Linux Foundation, v1.0 in 2026)
  models work as a **Task** whose `status.state` walks a lifecycle —
  `submitted → working → input-required → completed | canceled | failed` — with a
  human-readable `message`, streamed over Server-Sent Events for long-running
  work ([spec overview](https://a2a-protocol.org/latest/),
  [guide](https://galileo.ai/blog/google-agent2agent-a2a-protocol-guide)). That
  `state` is set **by the remote agent doing the work**. The client is told to
  believe it.
- **ACP** (IBM/BeeAI, REST-native), **ANP** (decentralized, W3C DIDs), and
  **AGNTCY**'s Agent Connect Protocol (Cisco/LangChain "Internet of Agents") each
  add discovery, identity, or a broker — and each still carries **status the
  producing agent authored**
  ([survey](https://arxiv.org/html/2505.02279v1),
  [taxonomy](https://arxiv.org/pdf/2606.19135)).
- The framework layer is the same shape one level down: **LangGraph** hands off
  through a shared typed state graph, **AutoGen/AG2** through GroupChat turns,
  **OpenAI's Agents SDK** through explicit control-passing handoffs
  ([2026 comparison](https://gurusup.com/blog/best-multi-agent-frameworks-2026)).
  In every one, the state a downstream node reads is the state an upstream node
  *wrote about itself*.

None of this is wrong — a shared vocabulary for task lifecycle is real progress,
and fak deliberately **interoperates** with it (it projects a policy-filtered A2A
Agent Card from its reviewed method registry; see the
[interoperability stance](../integrations/interoperability.md)). The point is
narrower: the wire format standardizes the *shape* of a status, not its *truth*.
Signed Agent Cards prove **who** an agent is; they do not make the agent's report
of **its own progress** any less self-authored.

The research frontier has converged on exactly this gap. "From Agent Traces to
Trust" argues agent trustworthiness "cannot be reduced to model accuracy alone;
it depends on whether the execution process can be reconstructed, inspected, and
governed" ([arXiv 2606.04990](https://arxiv.org/abs/2606.04990)); "VET Your
Agent" pursues host-independent autonomy via *verifiable execution traces*
([arXiv 2512.15892](https://arxiv.org/pdf/2512.15892)); "Right to History"
proposes a *sovereignty kernel* unifying Merkle audit logs and capability
isolation for verifiable agent execution
([arXiv 2602.20214](https://arxiv.org/abs/2602.20214)). The shared move: stop
grading the answer, start grading the **provenance**.

## fak's answer: a digest with no `claimed` field

fak's stance is structural, not exhortatory. It does not ask a worker to report
honestly; it hands a peer a status object **that has no field a self-report could
occupy.** The load-bearing property is a negative one: *the digest has no
`claimed` key by construction.* A peer reading it cannot pick up a self-report it
is never handed — `progress` is built only from the kernel's **verified** rung.

Three surfaces carry the same shape, at three grains:

### 1. The run digest — `dos_status`

`dos_status` folds four adjudicated verdicts about a whole run into one
peer-readable record: **liveness** (is it moving? — forward commit delta),
**progress** (read from the intent ledger's own rows, *never* the worker's
claim), **region** (the lease the run holds, so you know what files are unsafe to
touch), and **resume** (the plan, computed only once the run has stopped). Every
edge fails closed: a run with no intent ledger is a valid *zero-progress fact*,
not an error; a run holding no lease has an empty region. It is the legible,
peer-facing form of `dos verify`'s distrust discipline aimed at a run instead of
a single phase — and it deliberately returns **no `claimed` key**.

### 2. The task read — the A2A HTTP edge

`GET /a2a/v1/tasks/{id}` returns a task's own imperative record (`state`,
`result` — the forgeable, self-authored half) **and**, alongside it, a
`progress` object: the same no-`claimed`-field `VerifiedProgress` shape projected
across the edge ([`internal/gateway/a2a.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/gateway/a2a.go),
[`internal/relay/progress.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/relay/progress.go)).
`progress` is not a percentage and not a "done" boolean a closing leg could set;
it is a list of **steps**, each a durable `Ref` — a commit SHA, an issue `#1234`,
a memory slug, a path — that the reader can go re-read to confirm the step
happened. So a foreign peer polling the task gets, in the same payload, the
worker's story *and* a cursor into evidence the worker did not author.

### 3. The live correction — `a2achan`

The tightest case is one running agent correcting another mid-flight. In
[`internal/a2achan`](https://github.com/anthony-chaudhary/fak/blob/main/internal/a2achan/correction.go)
a worker exposes a live `WorkerStatus` row with a stable `Digest()`. An
orchestrator's `CorrectionRequest` is refused unless it **cites that digest** and
matches the worker's issue/task/lane exactly: a stale or fabricated status is
turned away as `UNWITNESSED` *before it reaches the worker's inbox*, an
out-of-scope correction as `TRUST_VIOLATION`, oversized text as `OVERSIZE`. And
the correction only counts as *witnessed* once the worker both **acks** it and
**reflects** its id in its next planned action — acked-and-reflected, not
acked-alone. Neither side is trusted on its word; each move must cite the other's
unforgeable handle.

Every one of these rides the same default-deny capability floor that gates an
ordinary tool call — coordination in fak is not a side library with its own
security surface (the full normative spec is the
[Multi-Agent Coordination Protocol RFC](../multi-agent-coordination-protocol.md)).

## Why it holds: progress is a cursor, not a number

The through-line: fak never gives a peer a place to write a self-reported status,
so a peer never reads one. Progress is a **re-verifiable cursor** — a pointer
into git and the intent ledger — and the reader re-derives the truth from the
artifact the worker could not forge. It is the live, agent-to-agent projection of
the repo's oldest rule: **the model proposes, the kernel disposes** — extended to
*the worker reports, the peer re-verifies.*

## Honest scope

- **The verified-progress cursor is wired but not yet fed on the HTTP edge.** The
  A2A task read projects `progress`, but no live `LedgerReader` is bound to the
  gateway task store yet, so `progress` fails closed to verdict `unknown` until a
  run's intent-ledger anchor is attached to a task. Wiring a file-backed reader
  onto the store is the named next rung — the seam and its contract test exist;
  the production feed does not.
- **The message bus is in-process today.** `a2achan`'s `InKernel` locale is real
  and shipped; the `Session` and `Window` locales share the identical code path
  keyed differently but lack durable cross-process backing. A crashed peer's
  mailbox does not yet survive the process — the durable-delivery work
  (outbox/ACK/redelivery) is tracked, not done
  ([#939](https://github.com/anthony-chaudhary/fak/issues/939),
  [#704](https://github.com/anthony-chaudhary/fak/issues/704)).
- **Some peer-state signals are still missing.** A peer cannot yet cleanly detect
  that another session is *mid-compaction* from the manifest
  ([#25](https://github.com/anthony-chaudhary/fak/issues/25)), and the dashboard's
  A2A awaiting list is not yet scoped per focused agent
  ([#2365](https://github.com/anthony-chaudhary/fak/issues/2365)).
- **No adoption claim.** This page explains a mechanism that ships in the repo; it
  does not assert who uses it. Against SOTA the distinctive move is the *negative
  space* — the absent `claimed` field — not a novel transport.

## Where to go next

- [Verify, don't trust](verify-dont-trust.md) — the after-the-fact sibling: the
  three things DOS re-checks once a worker stops.
- [RFC: the Multi-Agent Coordination Protocol](../multi-agent-coordination-protocol.md)
  — the normative spec under this explainer: message passing (`a2achan`), shared
  state (`sharedtask`), and wave primitives (`comm`), all on the capability floor.
- [Interoperability stance](../integrations/interoperability.md) — why fak
  projects onto A2A/ACP/ANP instead of shipping another SDK, with the honest
  per-wire grade.
- [The tool call is a syscall](tool-call-is-a-syscall.md) — the keystone the whole
  distrust discipline sits under.
