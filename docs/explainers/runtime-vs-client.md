---
title: "Two Runtimes, One Binary: Gateway vs Agent Runtime vs Client"
description: "\"Put fak in front of your agent\" hides three roles the one binary plays: the gateway runtime that governs model traffic (fak serve), the agent application runtime that hosts and runs the loop (fak serve --native), and the client that a harness becomes when you wrap it (fak manage). serve and guard are not alternatives or layers — they are a runtime and a client. This is the naming that unblocks the first decision: what do I even run?"
slug: runtime-vs-client
keywords:
  - agent runtime
  - AI gateway
  - inference gateway
  - agent application runtime
  - gateway runtime
  - fak serve
  - fak manage
  - runtime vs client
  - which do I run
  - agent SDK
  - managed agent runtime
  - two runtimes one binary
  - enforcement boundary
  - request arrival vs turn end
date: 2026-07-11
---

# Two runtimes, one binary — gateway vs agent runtime vs client

> **TL;DR:** the one `fak` binary plays three roles that "put fak in front of your
> agent" quietly conflates. **`fak serve`** is the **gateway runtime** — a long-lived
> server that governs *model traffic* (routing, cost caps, policy, quarantine, audit).
> **`fak serve --native`** is the **agent application runtime** — the same binary, but
> now *hosting and running the agent loop itself* (sessions, tools, subagents, streaming,
> resume), kernel-adjudicated. **`fak manage`** turns a harness you already run (Claude
> Code, Codex) into a governed **client** of a runtime. `serve` and `guard` are not two
> versions of the same thing and not two layers of a stack — they are a **runtime** and a
> **client**.

**Concept served:** the runtime/client naming spine (epic #3256, workstream A —
disambiguation). This is the page the rest of the agent-runtime story hangs on.

## Why this page exists

The onboarding line — *"point your agent at `fak serve`"* — works right up until someone
asks the obvious follow-up: **is `fak serve` an alternative to `fak manage`, a layer under
it, or its server half?** Nothing on the front page answers that, so the first real
decision — *what do I even run?* — stalls. Worse, "runtime" is doing double duty: the
thing that governs your **model calls** and the thing that runs your **agent loop** are
different servers, and the word alone doesn't tell them apart. This page names the three
roles so the choice becomes mechanical.

## The three roles

| Role | What it is | fak surface | Maturity |
|---|---|---|---|
| **Gateway (inference) runtime** | a long-lived server governing *model traffic*: model routing, cost/context caps, the default-deny capability floor, result quarantine, the audit journal | **`fak serve`** | shipped — the primary path today |
| **Agent application runtime** | a long-lived server that *hosts and executes the agent loop*: sessions, tool dispatch, subagents, streaming, resume — every step kernel-adjudicated | **`fak serve --native`** (drives the owned `agent.RunArm` loop) | shipping — the emerging first-class surface (workstream B) |
| **Client** | the harness or app that *calls* a runtime: Claude Code, Codex, or your own backend over an SDK | **`fak manage`** turns a harness into a governed client; or repoint one base URL yourself | shipped — witnessed live in front of Claude Code, opencode, Codex |

The load-bearing distinction is the first-vs-third row: a **gateway runtime** proxies the
model calls your existing harness makes and adjudicates the tool calls it proposes — your
harness still owns the loop. An **agent application runtime** owns the loop *itself*: it
is the server that decides the next step, dispatches the tool, spawns the subagent, and
streams the result, with the kernel on the call path for each one. `--native` is the flag
that flips `fak serve` from the first to the second — from *proxying someone else's turn*
to *running the turn itself*.

## One binary, drawn out

```text
  CLIENTS                        ONE fak BINARY                      UPSTREAM
  (call a runtime)               (plays two runtime roles)           (tokens)

  ┌──────────────────┐
  │ Claude Code      │           ┌──────────────────────────┐
  │ Codex / Cursor   │──repoint──▶│  fak serve               │
  │ your SDK backend │  base URL │  = GATEWAY runtime        │──▶ Anthropic / OpenAI
  └──────────────────┘           │  routes + governs model  │    Gemini / xAI /
        ▲                        │  traffic; adjudicates     │    Ollama / vLLM /
        │ fak manage              │  each proposed tool call  │    SGLang / llama.cpp
        │ wraps a harness        └──────────────────────────┘
        │ into a governed CLIENT              │
        │                                     │ add --native
        │                                     ▼
        │                        ┌──────────────────────────┐
        └────────────────────────│  fak serve --native      │
                                 │  = AGENT APPLICATION      │
                                 │  runtime: owns the loop   │
                                 │  (agent.RunArm), sessions,│
                                 │  tools, subagents, resume │
                                 └──────────────────────────┘
```

*The same static binary is the gateway runtime, the agent runtime (with `--native`), and —
via `fak manage` — the wrapper that makes your existing harness a governed client of it.
Nothing here is a second process or a separate install.*

## Which do I run?

| Your situation | Run | Which role |
|---|---|---|
| I already run Claude Code / Codex and want it cheaper + governed, no rewrite | `fak manage -- claude` | your harness becomes a **client**; guard starts a **gateway runtime** in-process |
| I have my own backend/SDK and want to govern its model calls | `fak serve` + repoint the base URL | **gateway runtime**, your code is the **client** |
| I want an always-on shared gateway for a fleet | `fak node` installs `fak serve` as a service | **gateway runtime** |
| I want fak to *host and run the agent loop*, not just proxy my harness | `fak serve --native` | **agent application runtime** |
| I just want to see the capability floor decide, no model | `fak preflight …` | neither — the offline adjudicator |

If you are not sure, you want the **client + gateway** path (`fak manage`): it is the
shipped default and changes nothing about how your agent is written. Reach for the
**agent application runtime** (`--native`) when you want fak to own the loop itself.

## The analogy you already have

Readers coming from the LLM-infra world already hold this split under other names:

- An **AI gateway** (Portkey, LiteLLM, an inference gateway) governs *model traffic* — that
  is the **gateway runtime**, `fak serve`.
- A **managed agent runtime / the Agent SDK** hosts and executes the *loop* — that is the
  **agent application runtime**, `fak serve --native`.
- The **app that drives them** is the **client** — for fak, a harness wrapped by
  `fak manage`, or your own SDK code.

fak's one contribution here is putting both runtimes behind **one static Go binary** with
the same kernel on the call path, so the client-vs-runtime and gateway-vs-agent-runtime
choices are flags, not separate stacks. → [One binary is the whole
surface](one-binary-one-surface.md).

## Honest scope

- **The gateway runtime is the mature path.** `fak serve` / `fak manage` is what is
  witnessed live today. Most adopters never leave the client-plus-gateway model, and that
  is the intended default.
- **The agent application runtime is emerging.** `fak serve --native` is real — it drives
  the owned `agent.RunArm` loop on the live serve path — but it is the newer surface being
  packaged into a first-class runtime under workstream B. Treat it as shipping, not
  settled, and prefer the gateway path unless you specifically want fak to own the loop.
- **"Client" is a role, not a product.** The same harness is a client of whichever runtime
  it points at; `fak manage` is just the one-command way to make it a *governed* one.

## Enforcement boundary — what a client gets on its own loop

The three roles answer *what do I run*; the next question an adopter with their own loop
asks is *what does fak actually enforce on MY loop, and what do I lose by not letting
`fak manage` own my process?* The axis that decides it is **request-arrival vs turn-end**:

> The gateway can refuse, throttle, reset, charge, or stop anything that **arrives** on
> the wire — trace-keyed, no harness required. It cannot compel a request your loop
> decided not to send. Everything welded to the **end of a turn** ("the model stopped —
> now what?") rides Claude Code lifecycle hooks, and those exist only when `fak manage`
> **owns the process** and installs them into the child's merged `--settings`.

Per mechanism, which boundary it fires on and whether it transfers to a bare
repoint-the-base-URL client:

| Enforcement | Request arrival (`fak serve`, over the wire) | Turn end (needs `fak manage` process ownership) | Anchor |
|---|---|---|---|
| **Per-turn admission** — run-state (pause/drain/stop) + turn/token budget, refused *before* the model runs | yes | — | `beginServedSessionTurn` (`internal/gateway/session_admit.go`), fed by the `decideSession` hook (`cmd/fak/main.go`) |
| **Hard spend cap** (#3273) — a scope over its versioned budget gets a kernel-side 409 refusal, never a plea to the model | yes — `spendBreach` refuses the turn, `chargeSpend` debits post-response, on any served path (serve or guard-as-client) | — | `internal/gateway/session_admit.go`, `internal/gateway/spend_governor.go`. *Caveat:* the governor must be attached via `SetSpendGovernor`; the `fak serve` CLI wiring is still open (#4859), so today attachment is programmatic, not a flag |
| **Pace shaping** — per-request min-gap delay and a max-tokens cap that can only lower what the client asked for | yes | — | `servedSessionTurn.minGapMs`, `maxTokensFor` (`internal/gateway/session_admit.go`) |
| **Budget / coherence session resets** — the human-like continue on token/context drain; a latched hard reset on the next admitted turn | yes | — | `isBudgetResetReason`, `armCoherenceReset` / `consumeCoherenceReset` (`internal/gateway/session_admit.go`) |
| **Operator control + steer** — the operator POSTs drain/stop or budget/pace changes; a steer lands at the *next* request boundary | yes | — | `steerSession` (`cmd/fak/main.go`) enqueues on the a2achan bus |
| **Confirm-token / reversibility gate** — an irreversible action needs a fresh operator token, checked per request | yes | — | `ReversibilityConfirmed` / `reversibilityGateVerdict` (`internal/adjudicator/decide.go`) |
| **Deny-all auto-continue** — resume the agent past a turn the floor refused entirely (the deny-all false-stop fix) | — | yes — Stop hook | `cmd/fak/guard_stophook.go`, installed in `cmd/fak/guard.go` |
| **Task-handoff gate** — a clean Stop must carry a `fak.task-handoff.v1` JSON (enforce for headless/fleet, auto-off attended-interactive) | — | yes — Stop hook | `cmd/fak/guard_handoff_mode.go` |
| **Operator-directed gate** — catches a headless turn ending by asking an absent human (#4951) | — | yes — Stop hook | `guardOperatorDirectedEffectiveMode` (`cmd/fak/guard.go`) |
| **PreCompact** — intercept the harness's context compaction | — | yes — PreCompact hook | `installGuardPreCompactHook` (`cmd/fak/guard.go`) |
| **Supervised restart loop** — on context-budget exhaustion, relaunch the child under the continuation trace with a carryover seed | — | yes — guard relaunches the child | `newGuardBudgetRestarter`, `--restart-on-budget` (`cmd/fak/guard.go`) |

Two honest footnotes, so no row outruns the code:

- **The residual weld is thinner than it looks.** Even the turn-end give-up decision is
  mostly gateway-side already: the counter it keys on — consecutive deny-all turns
  proposing the *identical* refused action — lives in the gateway
  (`denyAllSameSnapshot`, `internal/gateway/metrics_observe.go`). Only the *actuation*
  (actually ending or continuing the harness's turn) is harness-bound; exposing that seam
  to non-guard clients is #5161.
- **The per-session cache-break gate is inert today.** The counter, cause vocabulary, and
  Prometheus surface exist (`internal/metrics/cache_break.go`), but the #2915
  prefix-mutation detector that would produce live events never landed — do not read it
  as a working gateway signal.

So the practical answer to "what do I lose by not running `fak manage`": nothing in the
left column — admission, budgets, spend, pace, resets, steer, and the reversibility gate
all bind a bare base-URL client, because they fire when the request *arrives*. What you
lose is the right column: everything that reacts to your loop *stopping* — auto-continue
past a false stop, the handoff and operator-directed gates, PreCompact, and the
supervised restart — because the model-API wire never tells the gateway your turn ended.
This table is the shape future seam findings should land in: one row per mechanism, one
boundary, one anchor. (Driving tickets: #5152, #5153, #5161; captured by #5162.)

## Where to go deeper

- See both runtimes decide, offline, in one sitting: [Governed agent in 10
  minutes](../fak/governed-agent-quickstart.md) — one command for a governed session, then
  the server-side floor + audit + DENY path, no key or GPU.
- Point your existing agent at the gateway runtime: [Run your agent through
  fak](../integrations/README.md) and the [Claude Code guide](../integrations/claude.md).
- Why one binary carries both runtimes: [One binary is the whole
  surface](one-binary-one-surface.md).
- What the kernel does on every call, in either runtime: [The tool call is a
  syscall](tool-call-is-a-syscall.md).
- The vocabulary, one line each: [the fak/DOS glossary](glossary.md).
