---
title: "Fleet concepts: workers, lanes, leases, seats, and witnesses"
description: "What a fak fleet is: the control point, workers, accounts and seats, lanes and leases, waves, monitors, and witnesses — and where the public/private line falls."
---

# Fleet concepts

A fleet is a set of autonomous agent workers driven by one control point against one shared repository, coordinated so that no two workers edit the same region at the same time and no worker's claim of success is believed without independent evidence. The pieces below are the vocabulary the repository uses for that arrangement.

**Primary audience:** a reader who has met the word "fleet" in this repository and wants the pieces named before opening any runbook.

**The one-line disambiguation, because "fleet" names two different things here:** this page is about the **agent-worker fleet** — processes that resolve issues. A *box* fleet — GPU and worker machines — is a separate surface with its own page, [`fleetctl`](../fleet.md). When a document says "fleet" without qualification, check which of the two it means.

**Next action:** run the read-only status example in [Two safe commands](#two-safe-commands-you-can-run-now). Both are non-destructive and neither needs an API key.

## What problem does a fleet solve?

One agent working one issue at a time is bounded by wall-clock: a queue of a hundred issues takes a hundred sessions. Running a hundred agents at once against one shared trunk creates three problems a single agent never has, and a fleet is the machinery that answers them:

- **Collision.** Two workers editing the same files produce a merge mess or silently overwrite each other. Answered by *lanes* and *leases*.
- **Capacity.** Provider credentials, not ambition, cap how many agents can actually run. Answered by *accounts* and *seats*.
- **Credulity.** An autonomous worker that reports "done" is a self-report, and self-reports are wrong often enough to poison a backlog. Answered by *witnesses* and reconciliation.

A fleet is therefore less an execution engine than a trust-and-coordination arrangement wrapped around ordinary agent processes.

## How does work flow through a fleet?

The path from a filed issue to a closed one, with the coordination step that guards each hop:

```text
  issue / goal
       │
       ▼
  admission + routing        does this issue have a scope, a route, a witness plan?
       │                     which account and tier should serve it?
       ▼
  lane + lease acquired      claim a region of the tree; a conflicting claim is REFUSED
       │
       ▼
  worker spawned             one headless agent process, one issue
       │
       ▼
  commit on the trunk        the durable effect — uncommitted work is not shipped
       │
       ▼
  witness                    an independent check the worker did not author
       │
       ▼
  monitor + reconcile        close the issue, or return it to the queue flagged
```

The two hops that make it a *fleet* rather than a batch script are the lease (before the work) and the witness (after it). Everything else is a normal agent session.

## What is a worker?

A worker is one headless agent process assigned to one unit of work — in this repository, typically one GitHub issue. It runs unattended, commits its own result to the trunk, and exits. Workers are deliberately disposable: the fleet's durable state lives in commits, leases, and ledgers, never in a worker's memory.

The launch and rollup flow for issue-scoped workers is written down in [issue-scoped headless worker dispatch](../agentic-issue-dispatch.md); the always-on driver is [the issue-dispatch loop](../dispatch-loop.md).

## What is a lane?

A lane is a named region of the repository — a set of path globs — together with the queue of work that targets it. Lanes are the unit that arbitration reasons about: two workers in disjoint lanes may run concurrently, and two workers whose lanes intersect may not. Lanes also carry routing information, since the region a change lands in implies its work tier and its commit stamp.

## What is a lease?

A lease is a time-bounded claim on a lane's tree, taken *before* a worker starts editing. An overlapping claim is refused rather than queued, which is what keeps a shared trunk from becoming a collision domain.

Two honest details matter for anyone reading lease state:

- **Visibility is distributed; acquisition is not.** `fak leaseref` publishes lease records under the `refs/fak/locks/*` git ref namespace so a peer clone can *see* a lease after an ordinary fetch. The package documents itself as "DISTRIBUTION / VISIBILITY, not atomic acquisition" — the ref store makes a lease visible across machines, it does not make the grant globally atomic.
- **Liveness is judged by heartbeat, not by process id.** A lease's owning session is classified from a session-descriptor heartbeat, explicitly "never the ephemeral acquiring pid". A held lease whose acquiring subprocess has exited is normal, not stale.

## What is an account, and what is a seat?

An **account** is a named credential set for one provider — the unit the account switcher routes to. Two accounts can target the same provider, which is the point: you choose *whose* credential serves a given model. An account names the environment variable holding its key; the secret itself is never stored in the configuration file. See [the account switcher](../model-accounts.md).

A **seat** is one configuration-home identity a worker runs under (`fak accounts` is described as a "config-home registry: every `CLAUDE_CONFIG_DIR` seat with its disk-true identity"). Seats are the fleet's real capacity unit: a fleet cannot run more concurrent workers than it has usable seats, whatever the host's CPU count suggests. Seat inventory is one of the ceilings the capacity lens takes as input.

## What is a wave?

A wave is a set of work items that is safe to dispatch *simultaneously* because their lanes and expected paths are pairwise disjoint. Waves are computed by partitioning dispatchable leaves with the same disjoint-tree rule arbitration applies at launch — a first-fit graph colouring over lane and path overlap. The number of waves is therefore the number of sequential rounds a batch needs, and it is a property of the batch's collision structure, not a scheduling preference.

## What is a monitor or supervisor?

Monitors are the read-only surfaces that answer "what is the fleet doing, and what is stuck?" — worker health, stalled hosts, and what is currently gating progress. The fleet control surface is documented as **read-only by default**, with mutation kept behind explicit subcommands; separate verbs cover host stall fingerprinting and watchdog restart of dead monitors.

A supervisor differs from a worker in what it may touch: it reads fleet state and may replace or reap a dead worker, but it does not resolve issues itself.

## What is a witness?

A witness is an external check that the worker did not author, confirming a claimed effect. This is the fleet's answer to credulity, and the repository states it in exactly those terms: accepted witnesses include a passing parent-rerun test, a commit audit carrying a diff witness, an independent verification verb, or an independent read-back.

The distinction the vocabulary insists on is worth quoting directly: a worker saying it is done, or a local commit mentioning an issue number, is **not** the issue being closed. Only a reverified resolving commit is. The full vocabulary — `spawned`, `attempted`, `witnessed`, `closed`, `retried`, `stale` — is owned by [the dispatch SLO glossary](../dispatch-slo-glossary.md).

## Single agent or a fleet? A decision table

Pick the smallest arrangement that does the job. A fleet's coordination machinery is a real cost, and it earns that cost only past a certain queue size.

| | One agent session | A fleet |
|---|---|---|
| **Work shape** | One task you are watching | A queue of independent, pre-scoped units |
| **Coordination** | None needed | Lanes, leases, and wave partitioning |
| **Capacity limit** | Your attention | Provider seats |
| **Trust model** | You read the diff | Witnesses and reconciliation |
| **Failure mode** | You notice immediately | A silent unwitnessed claim poisons the backlog |
| **Worth it when** | Fewer than a handful of units | The queue is long and the units are genuinely disjoint |

The honest threshold: if you cannot state each unit's expected paths in advance, you cannot partition waves, and a fleet will spend its gains on collisions.

## Two safe commands you can run now

Both are read-only, need no API key, and print no host or credential identity.

**Read the currently visible leases.** This is the source that makes a peer's lease visible at admission:

```bash
fak leaseref live
```

Captured from this repository at the commit that added this page, while a resolver fleet held four lane leases — abridged to the first record. Note what a record contains: the lane, its kind, and the path tree it claims; never a host, user, or credential:

```text
[
  {
    "lane": "resolve-accounts",
    "lane_kind": "cluster",
    "tree": [
      "internal/accounts/**"
    ]
  },
  …
]
```

With no fleet running, the list is empty.

**Size a fleet before starting one.** The capacity lens is a dry-run calculator over Little's law; it launches nothing:

```bash
fak fleetcap --rate 40 --session 12 --seats 8 --json
```

Captured output, abridged to the assessment block:

```text
{
  "assessment": {
    "TargetRatePerHour": 40,
    "MedianSessionMinutes": 12,
    "ExactLoad": 8,
    "RequiredWorkers": 8,
    "AvailableWorkers": 8,
    "ShortfallWorkers": 0,
    "Verdict": "SUFFICIENT"
  },
  "target_rate_per_hour": 40
}
```

Read it as: sustaining 40 issues/hour at a 12-minute median session needs 8 concurrent workers, and 8 seats covers it. Raise `--rate` or lengthen `--session` and the verdict turns to a shortfall.

From a clone, substitute `go run ./cmd/fak` for `fak`.

## What is public, what is maintainer-only, and what is private

Three different things wear the word "fleet" in this repository, and conflating them is the most common misreading. The separation is a boundary, not a disclaimer.

| Layer | What it is | How to recognize it |
|---|---|---|
| **Public fak capabilities** | The governed gateway product: policy, adjudication, routing, and audit at the tool and model boundary. Fleet operation is not part of it. | Verbs tiered `[frontdoor]` in `fak help` |
| **Repository-maintainer operations** | The dispatch, lease, seat, and monitoring verbs this page defines. They exist to run *this* repository's backlog on a shared trunk. | Verbs tiered `[dev]` in `fak help` — the tiering line reads "`[frontdoor]` is the product; `[dev]` is `fak dev <verb>` tooling" |
| **Private lab infrastructure** | The control bridge that actually reaches lab machines, and the identities it carries. It is not in this repository. | Absent by construction; the seam is a per-box report JSON |

Every fleet verb named on this page is `[dev]`-tiered maintainer tooling. None of it ships as public product behavior, and nothing on this page should be read as a capability of a deployed `fak serve` gateway.

The public/private line for the *box* fleet is a data contract rather than a code import: the private bridge writes one report file per box, and the public tool reads, folds, and scores those files. Neither side imports the other, and the public tree names no host, channel, or token. That boundary and the gates that enforce it are documented in [the GPU-server private boundary](../gpu-server-private-boundary.md).

## Where the evidence lives

| If you want… | Read |
|---|---|
| **The worker launch and rollup runbook** | [Issue-scoped headless worker dispatch](../agentic-issue-dispatch.md) |
| **The always-on backlog driver** | [The issue-dispatch loop](../dispatch-loop.md) |
| **Witness, closure, and retry vocabulary** | [Dispatch SLO glossary](../dispatch-slo-glossary.md) |
| **What a running fleet currently reports** | [Dispatch status](https://github.com/anthony-chaudhary/fak/blob/main/docs/dispatch-status.md) · [fleet rollup](../fleet-rollup.md) |
| **Accounts, seats, and provider routing** | [The account switcher](../model-accounts.md) |
| **The box fleet (machines, not agents)** | [`fleetctl`](../fleet.md) · [fleet compute nodes](../fleet-compute-nodes.md) |
| **Where the private boundary falls** | [GPU-server private boundary](../gpu-server-private-boundary.md) |
| **Why the policy floor is the floor** | [Policy in the kernel](../explainers/policy-in-the-kernel.md) · [security model](../fak/security.md) |
| **Overloaded names, disambiguated** | [Product glossary](../glossary.md) · [contributor concept glossary](../fak/concept-glossary.md) |
| **Running a deployed gateway instead** | [Operator route](../operator/README.md) |
| **Every other documentation route** | [Documentation home](../index.md) |
