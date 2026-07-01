# Native Go harness — next GitHub steps, benchmarked against Crush

Date: 2026-06-30. Author: agent (Windows box; compute claims routed to fleet).

## The reference: what Crush is (`C:\work\crush`)

Charm's Crush is a mature, well-engineered Go coding agent. Its architecture is
the **proxy-loop** pattern done cleanly:

- **Loop:** `internal/agent/agent.go:550 sessionAgent.Run` — streams provider
  events (`OnReasoningStart/Delta`, tool-call accumulation at `agent.go:913-1068`),
  collects tool calls, executes them, appends results, re-enters. ~2200 lines.
- **Provider layer:** `charm.land/fantasy v0.33.2` — an **external SDK** owns the
  provider/turn mechanics. Crush orchestrates *around* fantasy; it does not own
  the model-call loop itself.
- **Sessions:** SQLite via sqlc (`internal/db`, `sqlc.yaml`) — durable session +
  message store, migrations, WAL-class persistence. **This is where Crush is ahead
  of fak.**
- **Permissions:** `internal/permission/permission.go:65 Service` — grant/deny/
  persistent, `Request(ctx, ...)` per call. An approval gate, not a kernel.
- **MCP:** `internal/agent/tools/mcp/*` — MCP tools/resources/prompts as first-class.

**The load-bearing contrast (the decisive finding):** Crush does **not own its
agent loop.** The send→get-tool-calls→execute→feed-back→repeat iteration lives
*inside* `fantasy.Agent.Stream` (called once at `agent.go:790`); Crush participates
only through **callbacks** (`PrepareStep`, `OnToolCall`, `OnToolResult`,
`OnStepFinish`, `StopWhen`) and a tool registry. Tool execution + the permission
prompt happen inside each `fantasy.AgentTool` closure. There is **no kernel seam** —
no single tool path the harness can intercept to materialize a result from cache
before execution, place a write barrier before consumption, or suspend a turn to
await an authoritative call. That is structurally impossible in a callback-streaming
SDK loop. **`fantasy` is exactly the layer fak's `RunArm` + kernel syscall boundary
replaces.** fak already owns what Crush delegates to a library. The next steps are
therefore not "catch up to Crush's loop" — they are "keep fak's kernel-owned loop
and borrow Crush's *host* seams (durable sessions, streaming, permission surface)."

## Where fak stands (evidence, not self-report)

The #1315 epic ("fak owns the native agent loop") + all 8 build-children
(#1316-#1323) are **CLOSED and diff-witnessed on disk**:

- Keystone #1316: `internal/gateway/native_serve.go:125` calls
  `agent.RunArm(ctx, s.planner, task, true, ...)` behind `fak serve --native`
  (commit `98bc30f3`, stamped `(fak gateway)`). Fork at `messages.go:159`.
- #1318 suspend/resume: `internal/agent/turn.go:53 Suspend` / `:69 Resume`, live
  in-package caller at `turn.go:175` (`s.spec.Predict`) + `:123`/`:189`.
- #1319 write barrier: `internal/abi/speculate.go` `Speculator.Predict`, now with a
  production caller (`turn.go:175`), default-off.
- #1317 consistency enum: `internal/abi/consistency.go` (STRICT..SPECULATIVE).
- #1320 REPL: `cmd/fak/chat.go:25 cmdChat` → `runChat` → `RunArm(..., true, ...)`.

Note: `dos verify 1315/1316` returns `shipped=false, source=none` — because the
referee keys on a `(fak <issue#>)` stamp and our convention stamps the *leaf*
(`(fak gateway)`), not the issue. The code shipped; the stamp grammar just doesn't
encode the issue number. Not a work gap.

## The one real gap: #1380 (OPEN) — the DoD witness run

The loop is **built and unit-tested but never witnessed owning a real run.** The
epic auto-closed on a *doc* commit-audit, which is not the DoD. #1380 is the single
open child and the true "next step":

> An AgentDojo (or small coding task) driven **entirely** by `fak serve --native`
> / `fak chat`, with ≥1 tool result materialized from vDSO before consumption
> (EngineCalls flat while VDSOHits increments) and ≥1 speculated turn that
> suspends, awaits the authoritative next call, and promotes/squashes within the
> same turn index (BufferSink.Rollback on mispredict). Artifact under
> `experiments/agent-live/`.

**Cannot run on this Windows box** (no model serving; `go test` OS-blocked). Must
run on **Mac Metal** (Qwen3.6-27B Q4_K, the witnessed inkernel path — `serve --metal`
+ `NewInKernelPlanner`) or a GPU node.

## Next GitHub steps (ordered, worst-regret-first)

### Tier 0 — close the epic honestly (the only thing blocking "done")
1. **#1380 DoD witness run** (OPEN, exists). Run on Mac Metal. Commit the artifact
   to `experiments/agent-live/`. This closes #1315 for real. Everything else is
   enhancement.

### Tier 1 — the two places Crush is actually ahead (file these)
2. **Streaming native serve** (NO issue yet — FILE IT). Today `messages.go:159`
   only takes the owned-loop fork for **non-streaming** `/v1/messages`; streaming
   still falls to the single-shot proxy (`gateway.go:1367`). Crush streams by
   default. Until the owned loop streams, the native path can't be default for
   interactive use. Wire `RunArm` onto `CompleteStream` (`stream.go:448`).
3. **Durable session state — adopt Crush's strongest idea.** #1363 (WAL turn
   checkpoint, OPEN) + #1365 (durable-by-default, OPEN) are fak's equivalent of
   Crush's SQLite session store, but fak's is unbuilt. This is the clearest
   "Crush does X better" gap. Prioritize #1363 as the durable-sessions keystone;
   it also de-risks the #1380 run (a killed native run should resume its exact turn).

### Tier 2 — make the owned loop the default front door (enhancement)
4. **`fak serve --native` on by default** once #1380 + streaming land — gated behind
   a doctor-visible durability posture (#1365).
5. **Permission/adjudication parity check** vs Crush's `permission.Service`: fak's
   kernel `Decide` (`gateway.go:1359`) is strictly more powerful (it's a kernel, not
   a prompt), but confirm the interactive `fak chat` REPL surfaces
   grant/deny/persistent the way Crush does — a UX gap, file if missing.

## The framing for any GH issue text

fak's native harness is not "another Crush." Crush is the best-in-class **proxy**
loop (SDK owns the model call; harness orchestrates). fak's thesis is **ownership**:
fak's own `RunArm` drives turns and the in-kernel syscall boundary is the *sole*
tool path — the precondition for vDSO-before-consumption, the write barrier, and
speculative suspend/resume, none of which a fantasy-style callback loop can do.
The next steps borrow Crush's **durability** (SQLite→WAL sessions) and **streaming
default**, while keeping the kernel seam that makes fak's loop categorically
different. Ship the witness (#1380) first — until it exists the program is PARTIAL.
