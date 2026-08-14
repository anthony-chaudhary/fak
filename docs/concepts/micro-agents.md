---
title: "Micro agents in fak: definition, lifecycle, and limits"
description: "A micro agent is one bounded agent loop fak hosts in-process. What micro agents are, when to use one, when not to, and how the kernel supports them."
---

# Micro agents

A micro agent is one bounded agent loop — a small task, a small context, and a fixed step budget — that a host process drives as a goroutine behind one shared kernel gateway, instead of running as its own operating-system process. Micro agents exist to make a fleet of agents cheap enough to run by the hundred: the unit of work stays an agent, but the unit of cost stops being a process.

**Primary audience:** readers deciding whether a micro agent is the right shape for a task, before opening any implementation code.

**Lifecycle:** the micro-agent host and the `fak micro` verb are a `gen/second-next` architectural **option**, not the default runtime. See [What is shipped and what is research](#what-is-shipped-and-what-is-research) below for the exact line, and [the generation contract](../generation.md) for what those labels mean.

**Next action:** run the no-key example in [Run one micro agent](#run-one-micro-agent-no-key-no-model-no-gpu). It needs no API key, no model weights, and no GPU.

## What is a micro agent?

A micro agent is an agent loop reduced to one interface with one method. In fak the contract is `Microagent.Step(ctx, gw) (done bool, err error)`: the host advances the agent by one unit of work — typically one model turn — and retires it when `Step` reports `done`. The agent never dials a model itself; it is handed the host's single shared gateway on every step.

> Source: `internal/microagent/microagent.go` — [`Microagent` and `Gateway`](https://github.com/anthony-chaudhary/fak/blob/main/internal/microagent/microagent.go).

Three properties follow from that shape, and they are what "micro" actually means here:

- **Bounded admission.** Spawns go through a bounded queue that refuses loudly (`ErrQueueFull`) rather than growing without limit.
- **Bounded concurrency.** Concurrent in-flight model calls are capped by a slot pool ("seats"), separate from the number of enrolled agents.
- **Bounded state.** Per-agent state is one session-table entry under a bounded LRU, and the whole host shares exactly one audit sink.

> Source: `internal/microagent/doc.go` — [package contract](https://github.com/anthony-chaudhary/fak/blob/main/internal/microagent/doc.go); `internal/microagent/slotsched.go` — [slot scheduler](https://github.com/anthony-chaudhary/fak/blob/main/internal/microagent/slotsched.go).

## Why use a micro agent?

The argument is density, and it is an argument about process weight rather than about model quality.

- **One goroutine per agent instead of two-or-more processes.** Production today runs roughly two operating-system processes per agent — a `fak manage` policy proxy wrapping an external agent CLI — plus one detached process per dispatch lane, each with its own audit journal. The in-process host replaces that with one goroutine per agent behind one gateway and one audit sink. *Witness:* `go test ./internal/microagent -run TestHostSmoke100AgentsOneGatewayOneAuditSink` runs 100+ agents in a single process against the real gateway server.
- **Intermediate tool chatter costs the orchestrator nothing.** An RPC subagent appends each intermediate call and result only to its **own** bounded context and hands back a bounded collapsed summary; the orchestrator pays context for the result, not the pipeline. *Witness:* `go test ./internal/microagent -run TestRPCSubagentCollapsesUnderFloor`.
  Dogfood the runnable spine with `fak micro collapse --calls 3 --payload-bytes 2048 --json`: the receipt reports `intermediate_tokens`, the non-zero standalone `folded_tokens` estimate for the result the parent admits, `saved_tokens`, and one journal row per governed child call. This is a deterministic context-capacity witness, not a provider-billing claim.
  The first real project workflow is `fak micro collapse repo-pulse --dir . --json`: one child performs the recurring status/head/diff reads and returns a bounded pulse, while the receipt compares three inline parent tool turns with one collapsed turn. `tool_turns_skipped` and token fields are explicit counterfactual estimates, not provider usage.
- **Parked agents stop costing RAM.** Hibernation freezes a parked agent's context to disk at a step boundary and releases its goroutine, restoring the context byte-identically on wake — so a host can enroll N agents while keeping only R resident.

> Source: [`rpcsubagent.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/microagent/rpcsubagent.go) · [`hibernate.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/microagent/hibernate.go) · [`microagent_test.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/microagent/microagent_test.go).

## When should I not use a micro agent?

Prefer something else in each of these cases. The first four are stated limits of the current implementation, not style preferences.

- **The task is a single deterministic tool call.** Use `fak preflight` and get one adjudicated verdict. An agent loop buys you nothing when there is no loop.
- **The agent needs per-agent operating-system state.** Its own working directory, credential files, or process-tree tools do not fit a goroutine-per-agent model. The package names this as its own invalidating assumption: if the real loop cannot be stepped without per-agent OS state, the host under-isolates and needs a subprocess execution seam first.
- **You need a real provider today.** Only the Mock engine is supported in-process by this path; `fak micro --engine` accepts nothing else.
- **The agent must hibernate while blocked inside a live tool call.** Go does not serialize a goroutine stack, so hibernation happens only at a step boundary — between units of work, never mid-`Step`.
- **Provider seats, not process weight, are your binding cost.** That is the package's stated demotion criterion: if per-agent cost is dominated by rate limits rather than local process weight, the in-process host buys no density and should be retired.

For the full-agent route instead, read [the managed agent runtime](../explainers/agent-runtime.md).

## How does fak support micro agents?

Three surfaces, at three levels of maturity.

1. **A CLI verb.** `fak micro` runs one micro agent end to end; `fak micro host` boots the in-process host and runs a small fleet; `fak micro trace <id>` prints one agent's structured timeline. Config precedence is flags > env (`FAK_MICRO_*`) > file > defaults, the same precedence the rest of the binary uses.
2. **A host package.** `internal/microagent` supplies the worker pool, bounded spawn queue, slot scheduler, hibernation, fair scheduling, retry, and the single audit sink.
3. **A capability floor.** Every tool-execution backend — the in-process goroutine function, the subprocess backend, and any container or remote backend registered later — executes only behind the in-process kernel adjudication floor. The seam adjudicates *before* dispatch, so a denied call costs zero execution at every isolation level, and no package API yields a bare unadjudicated executor. *Witness:* `go test ./internal/microagent -run TestAdjudicationFloorBlocksDeniedActionAcrossAllRegisteredBackends`.

**The honest caveat, because it changes what the CLI proves:** the `fak micro` run path drives the deterministic Mock planner as its gateway seam. The served kernel gateway and the tool-execution floor are **not yet wired into that verb**; the Mock run bounds concurrency through the slot pool only. The `--admission-*` caps are resolved and reported by `fak micro` but enforced on the served gateway, not by the Mock run. So the floor invariant above is witnessed by the package's conformance suite — not by the command you type.

> Source: `cmd/fak/micro.go` — [the `fak micro` front door](https://github.com/anthony-chaudhary/fak/blob/main/cmd/fak/micro.go), whose header states the same caveat. Policy background: [Policy in the kernel](../explainers/policy-in-the-kernel.md) and [the security model](../fak/security.md).

## The micro-agent lifecycle

One agent moves through four states, and the host owns all four:

1. **Spawn** — the agent is accepted onto a bounded queue, or refused loudly when the queue is full.
2. **Step** — a worker drives `Step` repeatedly. Each model call first acquires a seat from the slot pool, so concurrency is capped independently of how many agents are enrolled.
3. **Retire** — the agent leaves the loop on `done`, on an error, or on cancellation. Each transition emits one audit event (`spawn`, `reject`, `done`, `cancel`, `retry`, `verify`, `error`).
4. **Reap** — the host reclaims the goroutine and the agent's slot. `Drain` retires the fleet gracefully; `Close` cancels it.

A parked agent between steps may be hibernated to disk instead of holding a goroutine, and woken byte-identically later.

## Run one micro agent

### Offline smoke path (no key, no model, no GPU)

Resolve the plan first without spending anything:

```bash
fak micro --dry-run
```

Captured output, from this repository at the commit that added this page:

```text
fak micro run — resolved plan (dry-run — no spend)
  engine:                    mock
  isolation backend:         goroutine   (registered: goroutine, subprocess)
  seats (slot pool):         0   (effective: 8)
  host workers (K):          8
  spawn queue:               256
  agents (N):                1
  turns/agent:               1
  admission max-concurrent:  unbounded   (M19; enforced on the served gateway)
  admission token-budget:    unbounded   (M19; enforced on the served gateway)
  precedence:                flags > env (FAK_MICRO_*) > file > defaults
```

Now actually run one agent end to end:

```bash
fak micro
```

```text
fak micro run — 1 agent(s) on mock, slots=8
  micro-000    done  (1 step(s))
  spawn=1 done=1 error=0 cancel=0  |  1 done / 0 failed
```

Run three agents for two turns each, persist their traces, then read one agent's timeline back:

```bash
fak micro -n 3 --turns 2 --trace-out traces.jsonl --json
fak micro trace micro-000 --trace-in traces.jsonl
```

```text
trace micro-000 — 6 span(s), 58 token(s)
  outcome: success
  verdicts: ALLOW
  #0   seat      acquire  (seat=slot-pool/8)
  #1   step      turn 1  (tokens=29)
  #2   verdict   mock-planner  (verdict=ALLOW)
  #3   seat      acquire  (seat=slot-pool/8)
  #4   step      turn 2  (tokens=29)
  #5   verdict   mock-planner  (verdict=ALLOW)
```

The trace makes the lifecycle above literal: one seat acquisition per turn, one planner verdict per turn, and a terminal outcome. The `mock-planner` verdict label is the honest one — this run is the deterministic Mock engine, not a provider and not the served kernel gateway.

From a clone, substitute `go run ./cmd/fak` for `fak`. The regression that pins this behavior is `go test ./cmd/fak -run TestMicroRunEndToEndOnMock`.

### Real fak-kernel gateway path

Start the kernel, then point the in-process host at it explicitly:

```bash
# terminal 1 — the actual fak policy/session/cache gateway
fak serve --addr 127.0.0.1:8080

# terminal 2 — two goroutine microagents, one shared session table and gateway
fak micro --engine gateway --gateway 127.0.0.1:8080 --model kernel-model \
  --agents 2 --workers 2 --seats 1 --turns 1 --trace-out micro-real-kernel.jsonl --json
```

`--gateway` is normalized to the OpenAI-compatible `/v1/chat/completions` endpoint. Every agent turn carries its agent id into one `microagent.SessionGateway`, which gates and debits one shared `session.Table` around the HTTP call. One cooperative scheduler still bounds calls across the whole host. The kernel may use its deterministic no-upstream planner, a configured provider, or a local model; `engine=gateway` claims only that the microagent crossed the real fak kernel seam, not that weights or a paid provider were involved.

The wire regression is `go test ./cmd/fak -run TestMicroGatewayEngineUsesRealFakKernel`: two agents must make two requests to `/v1/chat/completions`, preserve the requested model, and record provider-reported usage in separate timelines. `TestSessionGatewayConcurrentLoadOneTableRaceClean` separately pins concurrent fan-in through one session table.

## Micro agent vs. full agent vs. subagent vs. tool call

Four shapes, ordered from most autonomy to least. Pick the least autonomous one that does the job.

| | Full autonomous agent | Micro agent | Subagent | Deterministic tool call |
|---|---|---|---|---|
| **What it is** | A long-lived agent loop with an open-ended goal | One bounded agent loop with a small task and a step budget | A micro agent spawned *by* an orchestrator to collapse a pipeline | One adjudicated effect, no loop |
| **Who drives the loop** | The agent itself, until it stops | The host's worker pool, one `Step` at a time | The parent, via `RunScript` | Nobody — a single call |
| **Context** | Grows across the whole session | Bounded per agent | Bounded, and **separate** from the orchestrator's | One request and one verdict |
| **Runtime unit** | Its own process (a `fak manage` proxy plus an agent CLI) | A goroutine behind one shared gateway | A goroutine plus a collapsed summary | A function call behind the policy floor |
| **Try it with** | `fak manage -- claude` · `fak agent --offline` | `fak micro` · `fak micro host` | `internal/microagent` `RunScript` (no CLI verb yet) | `fak preflight --tool <name> --args '{}'` |
| **Maturity** | Shipped, default path | `gen/second-next` option, Mock engine only | Package-level spine, test-witnessed | Shipped, default path |

The row that matters for a decision is *runtime unit*. Everything else about a micro agent — the bounded queue, the seats, the hibernation — exists to make that row's "goroutine" affordable at fleet scale.

## What is shipped and what is research

Scope labels here follow [the generation contract](../generation.md). Read this section before quoting anything above as product behavior.

- **Shipped, default path.** `fak manage`, `fak serve`, `fak preflight`, and the dispatch spawn path. These are what runs when you do not type `fak micro`.
- **Shipped as explicit options, `gen/second-next`.** `fak micro` constructs the `internal/microagent` host. Its default remains deterministic Mock; `--engine gateway --gateway HOST:PORT --model ID` sends every turn through one running fak kernel and one shared session table. `dispatch tick --backend micro` is a separate offline enrollment prototype, not evidence of provider-backed issue resolution.
- **Test-witnessed package spine, no CLI surface.** The RPC subagent collapse and the adjudication-floor conformance across registered execution backends. Real code with real tests; not a command you can run.
- **Research, not product behavior.** The micro-*context* fabric work — splitting one cached agent base into many bounded logical contexts at 100 / 1,000 / 10,000 scale — is a research ladder with dated, scoped witnesses. It is adjacent to micro agents and often confused with them: micro *agents* are loops, micro *contexts* are the bounded slices of cached state those loops could run over. Start at [micro-context fabrics](../research/micro-context-fabrics.md) and treat every claim there as a hypothesis or a scoped observation until a maintained authority adopts it.

Promotion and retirement for the micro-agent host are written down rather than implied: the shared-kernel call seam is now runnable, but promotion to a default issue-resolution backend still requires the paired quality/cost/wall-clock evidence in #2028/#6520. Retire it if that measurement shows provider seats dominate, or if the isolation floor demands per-agent operating-system processes anyway.

## Where the evidence lives

| If you want… | Read |
|---|---|
| **The host contract, promotion and demotion criteria** | [`internal/microagent/doc.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/microagent/doc.go) |
| **The `fak micro` verb and its stated caveats** | [`cmd/fak/micro.go`](https://github.com/anthony-chaudhary/fak/blob/main/cmd/fak/micro.go) |
| **The subagent collapse spine** | [`internal/microagent/rpcsubagent.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/microagent/rpcsubagent.go) |
| **The execution-floor conformance suite** | [`internal/microagent/toolexec_floor_conformance_test.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/microagent/toolexec_floor_conformance_test.go) |
| **Why the capability floor is the floor** | [Policy in the kernel](../explainers/policy-in-the-kernel.md) · [security model](../fak/security.md) |
| **The full-agent route instead** | [Managed agent runtime](../explainers/agent-runtime.md) |
| **Micro-context research and its maturity rules** | [Research route](../research/README.md) · [micro-context fabrics](../research/micro-context-fabrics.md) |
| **What fak claims versus what it ships** | [Claims ledger](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) |
| **Every other documentation route** | [Documentation home](../index.md) |
