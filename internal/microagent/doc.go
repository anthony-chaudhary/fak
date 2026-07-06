// Package microagent hosts many agent loops in ONE process: a worker pool that
// drives K concurrent Microagent.Step calls as goroutines, all sharing one
// in-process kernel gateway (#2002, epic #2000 M2).
//
// Production today runs ~2 OS processes per agent — a `fak guard` policy proxy
// wrapping an external CLI — plus one detached process per dispatch lane
// (cmd/fak/dispatch_tick.go spawn path), each with its own hash-chained audit
// JSONL. This package is the in-process alternative: one goroutine per agent,
// ONE shared gateway seam (Gateway = agent.Planner, the seam internal/gateway.New
// builds exactly once for all served sessions), per-agent state as ONE
// session.Table entry per agent (bounded LRU, limit 8192 — internal/session),
// and ONE AuditSink for the whole host. Lifecycle is explicit: Spawn (bounded
// queue, loud ErrQueueFull) → Step loop → retire → Reap, with Drain (graceful)
// and Close (cancel) on the host and Cancel on one agent.
//
// The matching minimal registration set is internal/registrations/microagent
// (#2009): a host process blank-imports that instead of the full defconfig; the
// smoke test here does exactly that and runs 100+ agents against the real
// internal/gateway server over the Mock engine.
//
// Tier: integrator (4) — see internal/architest. This package may import only
// packages whose tier is <= 4; an upward import fails the architest gate.
//
// Generation intent: gen/second-next architectural exploration (#2002). This is
// an OPTION behind an explicit import boundary — nothing in the default
// `fak serve`/`fak guard`/dispatch path constructs a Host. Closing evidence for
// the generation frame:
//
//   - Promotion evidence: the smoke test (microagent_test.go) witnesses >=100
//     microagents completing in one process behind exactly one gateway and one
//     audit sink, with one session-table entry per agent. Promote once the
//     dispatch path can target the host instead of detached CLI processes
//     (#2030) and a density/$-per-task measurement (#2033) confirms the
//     per-agent process weight was the binding cost.
//   - Demotion / retirement criteria: retire the host if the #2033 benchmark
//     shows per-agent cost is dominated by provider seats/rate limits rather
//     than local process weight (the host then buys no density), or if the
//     isolation floor demands per-agent OS processes anyway (#2018 failing to
//     hold the in-process adjudication floor at this isolation level).
//   - Invalidating assumption: Step assumes an agent loop can be driven as a
//     resumable in-process step function over a shared planner seam. The #2001
//     extraction (per-turn stepping of internal/agent RunArm) is still open —
//     if the real loop cannot be stepped without per-agent OS state (working
//     directory, credential files, process-tree tools), the goroutine-per-agent
//     model under-isolates and this host must grow a subprocess ToolExec seam
//     (#2003/#2014) before it can carry production agents.
package microagent
