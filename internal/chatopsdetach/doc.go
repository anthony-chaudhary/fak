// Package chatopsdetach is the pure detached-execution decision kernel for
// chatops ACT verbs (#2265, epic #2259 leaf C5): the fold that turns an inbound
// "dispatch this" command into ack-now / witnessed-completion / stall-escalation
// routing, without ever touching the wire, a git repo, or a clock.
//
// Tier: foundation (1) — see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
//
// # The seam
//
// The chatops control surface has two halves. The READ half answers queries. The
// ACT half — dispatch a run, resume a stalled loop — must DETACH: the slow work
// runs out-of-band and the channel gets an immediate ack, then a witnessed final
// edit when the run lands. This package is the ACT half's decision core. The
// inbound door (internal/chatops, epic leaf C4, not yet landed) parses a message
// into a Command; the guarded-dispatch front door (internal/dispatchtick's
// EvaluatePreflight) decides admission; the durable card + outbox
// (internal/slackoutbox, internal/dispatchpost) carry the ack and the witnessed
// finalize; the blockers surface (internal/blockerpost) carries a stall page.
// chatopsdetach is the pure fold BETWEEN those impure edges: (command, admission
// verdict, prior spool row) in, decision out.
//
// Keeping it pure is the point — the same discipline internal/dispatchtick's
// preflight and dos_arbitrate hold. State in, decision out, no I/O, so a replay
// of the same inputs yields the same decision and a test needs no Slack, no git,
// and no wall clock. The impure shell owns the run, the spool file, and the send.
//
// # The three properties it decides
//
//  1. Idempotent dispatch. Slack has no server-side idempotency, so a dropped ack
//     or a retried event can deliver the same Command twice. Decide keys on the
//     command Nonce against a durable spool Record: a re-delivery of an already
//     dispatched nonce re-acks the SAME run (ReAck) and starts NOTHING, so a
//     double delivery can never double-dispatch. The ack text is deterministic,
//     so the re-ack is byte-identical to the first and coalesces on the outbox.
//
//  2. Refusal at admission. When the front door refuses (full seats, no lane
//     seat, gate pressure), Decide routes a Refuse carrying the closed refusal
//     token — it never mints a run and never dispatches, so a refused command
//     cannot silently queue-jump the cap. A refusal writes no terminal spool row,
//     so a later delivery is re-admitted honestly when capacity frees rather than
//     frozen on the first "no".
//
//  3. Stall escalation. A detached run that goes silent past its budget is judged
//     out-of-band (JudgeStall) into a blockers escalation — a background "status"
//     note past one budget, an "operator" page past PageMultiple budgets — read
//     from witnessed liveness, never the worker's self-report.
//
// # What it is NOT
//
// There is no inbound listener, no message parsing, no Slack client, and no run
// launcher here. This kernel does not decide admission (that is the preflight's
// closed-vocabulary job); it ROUTES an already-made admission verdict. It never
// mints a run id or steals a seat. Completion is not narrated: the witnessed
// finalize rides slackoutbox.Card's Witness, which derives SHIPPED from evidence,
// never from a text claim. This package only decides what the shell must do next.
//
// # Invariants and Guards
//
// Invariant: chatops detach decision logic is fail-closed and deterministic.
// No run can be dispatched without an explicit Admitted verdict from the front door.
// Guard: prior dispatched records strictly take precedence over fresh admission to prevent duplicate execution.
package chatopsdetach
