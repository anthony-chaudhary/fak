# Native-harness progress tracking (#1315)

Date: 2026-06-29

The roll-up for [epic #1315](https://github.com/anthony-chaudhary/fak/issues/1315) —
**fak owns the native agent loop end-to-end, the dispatch-edge program**. The forward
framing lives in
[the vToolcall fork note](VTOOLCALL-FORK-AND-BESTEFFORT-2026-06-25.md);
this note is the per-surface *evidence map* that says, with `file:line` witnesses, what is
SHIPPED on the live serve path, what is built-but-unwired (PARTIAL), and what is genuinely
missing (NOT_YET).

It was produced from an 8-surface, 90-claim survey whose every SHIPPED/PARTIAL claim was
re-checked by an independent adversarial pass against git history and request-path
reachability — **27 claims were downgraded** on the one question that matters here: *is
there a live, non-test caller on the request path?* The repo's dominant failure mode is
shipped-but-unwired, and it is concentrated in exactly this program.

## The thesis

On the proxy path the external harness (Claude Code, codex) owns the turn loop and
*consumes* tool calls outside fak — fak can deny, repair, or quarantine a proposed call,
but it cannot synthesize a tool result as if it ran, suspend a turn to await a provisional
result, or place a write barrier before consumption. The native harness is fak *owning*
dispatch: fak's own loop drives the turns and the kernel is the only tool path. That
ownership is the precondition for every "impossible on proxy" capability — vDSO
tool-result materialization, before-consumption write barrier, speculative serve.

The honest one-liner: **the engine is owned, the loop is not (yet).**

## Owns today (SHIPPED, live on the serve path)

The model-side and adjudication-side foundation is real and on the live serve path. What is
missing is loop *ownership*, not the pieces.

| Capability | Witness |
|---|---|
| Native `/v1/messages` front door | `internal/gateway/http.go:74` -> `handleAnthropicMessages`; decode `internal/agent/anthropic_server.go:89` |
| In-kernel generation (`--gguf`, no `--base-url`) | `internal/gateway/gateway.go:757 NewInKernelPlanner`; `gateway.go:1254 s.planner.Complete`. Witnessed Mac Metal Qwen3.6-27B |
| In-kernel tool-call render + lift | `inkernel_render.go:49 renderChatMLTools`; `inkernel_planner.go:684 normalizeCompletionToolCalls`; `TestInKernelForwardEmitsLiftableToolCall` |
| Tool-call adjudication on the served turn | `messages.go:637 adjudicateProposed` -> `gateway.go:1359 k.Decide`; e2e `inkernel_multiturn_toolloop_wire_test.go:75` |
| Fail-closed on malformed tool calls | `inkernel_planner.go:690 ToolCallsDropped`; 502 at `http.go:496` |
| vDSO read-serve before adjudication | `kernel.go:353-360` (no engine dispatch); `fak_read` MCP tool wires it (`mcp.go:446`) |
| Session admission per served request | `session_admit.go:53 beginServedSessionTurn`; budget debit `gateway.go:1300` |
| Stateless + persistent context planner | `gateway.go:1078 maybePlanMessages`, default-on at `--ctx-view-budget=8000` |
| KV planned-elision residency bridge | `agent.KVSpanElider.ElideKVSpans` behind `FAK_INKERNEL_KVMMU` (bit-exact, max\|delta\|=0) |
| Inbound result quarantine + tool-def pruning | `messages.go:600 admitInboundResults`, `messages.go:515 maybeCompactInboundTools` |
| `fak guard` proxy gate | wraps any harness on a private loopback, adjudicates every call (`guard.go:505`). The gate we keep; the native loop complements it |
| In-kernel syscall boundary + A/B comparator | `kernel.Syscall` (`kernel.go:579`), `execViaKernel` (`loop.go:141`); `agent_test.go:12 TestOfflineABTurnDelta` |

## Partial (engine exists, harness doesn't own it yet)

Each of these is fully built and tested, and each has **zero live caller on the request
path**. This is the honest middle, and the bulk of #1316's payoff is converting it to LIVE.

| Capability | Built at | Why PARTIAL |
|---|---|---|
| `RunArm` multi-turn loop | `loop.go:244` | Only callers: `dojo.go:372` (bench), `main.go:607` (`fak agent` benchmark), `internal/ablate`. Served path is single-shot `s.planner.Complete`, never `RunArm` |
| Session-control gate (`WithSessionTable`/`gateTurn`) | `loop_session.go:41,101` | No production caller passes the table |
| Operator steer splice (`drainSteer`) | `loop_session.go:133`, spliced `loop.go:280` | Producer wired (`POST /v1/fak/session/{id}/steer`); consumer is RunArm-only; gateway never drains the bus |
| Per-tool-call routing (`WithRouteManifest`) | `loop_session.go:56` | `agent.Run` never passes it (proxy-side per-call routing IS live at `gateway.go:1547`) |
| `ApplyPace` budget scaling | `ctxplan_session.go:114` | Zero non-test callers |
| Pause/resume warm-KV splice (`WaitResume`) | `resume.go:103`, `warmsplice.go` | Zero callers outside `internal/session`+tests; `gateTurn` *terminates* on Paused |
| Streaming in the loop | proxy `messages_stream_planner.go:154` | `RunArm` calls buffered `p.Complete` only |
| AgentDojo full-stack defense (ASR=0) | artifact `experiments/agent-live/...` | `register.go:16` self-documents "nothing in the boot path drives `abi.Stewards()`" |
| Multi-session scheduler (`Table.Pick`) | `scheduler.go` | Zero serve-path callers; `NativeScheduler` is a shape-proof stub |
| Intent conduit (`TurnIntent`, #805/#809) | `session.go:274` | Never called in production; `NativeScheduler` does not consume intent |
| Loop lifecycle ledger | `bgloops.go:31` | `bgloop.New()` called without observer/admit options; `rsiloop`/`shipgate` offline |

## The gap (NOT_YET)

The first two are the keystones — nothing in the speculative half runs until they land.

- **Suspend-and-resume turn primitive (#1318).** No `Turn.Suspend -> ProvisionalSink` /
  `Turn.Resume -> abi.Outcome`. `gateTurn` *terminates* the arm on a non-proceed verdict
  (verified `loop_session.go:101`); `WaitResume` is whole-session, not turn-granular, and
  uncalled. The one net-new mechanism the canonical note names.
- **Before-consumption write barrier (#1319).** The foundation is shipped-but-dormant: vDSO
  read-serve is live, and SEAM-4 (`abi/types.go:114 SpeculationContext`/`Outcome`/
  `ProvisionalSink`, `abi/speculate.go NewSpeculator`/`BufferSink`/`specEffectFree`,
  committed `d859ec21` #812) is committed and tested — but `NewSpeculator`/`Speculator.Predict`
  have **zero production callers** (only the definition + tests; the `r.Predict` in
  `internal/dojo/claims.go:135` is an unrelated dojo predictor). There is no suspension
  point for a speculation to hang on until #1318 exists.
- **Consistency-level field (#1317).** abi carries a `WorldVer` clock but **no
  `ConsistencyLevel` on `ToolCall`** (grep-confirmed). Thread
  STRICT/BOUNDED_STALE/BEST_EFFORT/SPECULATIVE before claiming any best-effort serve.
- **Native operator front door (#1320).** No interactive agent REPL (`cmdTUI` is a loops
  console; `cmdAgent` is a one-shot benchmark). The Apache-clean single-binary native build
  the note prescribes is unbuilt.
- **System-prompt MMU spine (#1322).** `internal/syspromptmmu` Rungs 1-5 are committed with
  tests but have **zero external callers** (the only tree hits are comment references in
  `capindex/catalog.go` and `promptmmu/promptmmu.go`). No request-path spine.

## Relationship to existing epics

| Epic | Relationship |
|---|---|
| #1193 session-lifecycle | Dependency — owned loop consumes `session.Table` on the live path; reuses `WaitResume` at turn granularity |
| #748 agent-OS | Dependency / parent — owned loop is the PCB scheduler body; the TUI is its operator console |
| #809 speculative agent-loop | Child — the suspend/resume primitive IS its missing mechanism; the write barrier is its continuation |
| #844 ctxplan reachability | Overlap — `RenderTurn` live on proxy; `ApplyPace` composition is the unfulfilled half |
| #1173 verified loop | Dependency — `loopgate.Adjudicate` live for the loop container; this extends it to the loop body |
| #1258 system-prompt-MMU | Child — supplies Rungs 1-5; this epic provides the first live request-path caller |
| #1103 skill-loader query | Overlap — the overlay's content source once the loop authors the system block |
| #1206 guard --resume | Twin — the proxy-side resume; the in-loop resume is its native counterpart |
| #805 intent conduit | Child — `TurnIntent` flows into placement only once a loop fak owns produces it |

## Ordered children

Worst-regret-first, dependency-first. Keystones are #1316 (a live caller for the loop) and
#1318 (the one net-new primitive).

1. #1316 ⭐ host `RunArm` as the live serve loop behind `fak serve --native`
2. #1317 thread the consistency-level field through `ToolCall`
3. #1318 ⭐ add the suspend-and-resume turn primitive
4. #1319 turn the before-consumption write barrier live
5. #1320 minimal native TUI/REPL on the `internal/agent` seam (`fak chat`)
6. #1321 wire session control end-to-end on the native loop
7. #1322 author + query the system-prompt overlay from the owned loop
8. #1323 per-turn harness-quality gate (loop-body witness)

## Definition of done

One witnessed demonstration: an **AgentDojo (or coding-task) run driven entirely by
`fak serve --native` / `fak chat`** — fak's own `RunArm` owns dispatch, the kernel is the
sole tool path, no external harness owns the loop — in which **at least one tool result is
materialized from vDSO before consumption** (engine-dispatch counter flat for that call)
**and at least one speculated turn SUSPENDS, awaits the authoritative next call, and
promotes or squashes within the same turn index**. The witness artifact must show: a grep
proving `RunArm` was called from a non-test serve-path caller; the per-turn `ArmMetrics`
(`Turns`, `VDSOHits`, `Repairs`, `Quarantines`) on the served response; a speculative-serve
counter > 0 with a matching `BufferSink.Rollback` on a mispredict; and ASR=0 carried with
`fak_modified=true` honestly labeled.

Until that single run exists, the program is PARTIAL. Honesty discipline per
[the epic close-out method](EPIC-CLOSEOUT-METHOD-2026-06-29.md): a child's
CLOSED box is a self-report, not a witness — close a child only when the artifact is on
disk and the claimed seam is reachable from a non-test caller.
