// Package toolproc is the kernel's process table for tool calls — the lifecycle
// spine for LONG-RUNNING tool use.
//
// THE GAP IT CLOSES. fak's thesis is "treat the tool call like a syscall: the
// model proposes, the kernel disposes." Today the kernel disposes at exactly two
// instants: admission (abi.Adjudicator, before dispatch) and result admission
// (abi.ResultAdmitter, before the payload re-enters context). Between those two
// instants a tool call is invisible — fine when every call was a sub-second
// request/response, wrong now that harnesses run background shells, monitors,
// subagents, and remote jobs: tool calls that live minutes-to-hours, emit
// streams, and outlive their turn. A `Bash(run_in_background)` launch, the job
// it spawns, and the later `BashOutput` polls cross today's floor as
// INDEPENDENT, UNCORRELATED events; nothing models "this call is still
// running", so nothing can enforce a runtime deadline, detect a stall, reap an
// orphan, or refuse a completion that arrived after its kill. A kernel that
// gates entry but does not own the process table is a doorman, not a kernel.
//
// THE OBJECT. A tool call that goes long becomes a tool PROCESS: a kernel
// object with an owner (session), a declared runtime envelope granted at
// admission (deadline + heartbeat cadence), observed liveness signals (pulses:
// heartbeats, output chunks, progress, polls — each optionally correlated back
// via the polling call's TraceID), and a terminal transition (exit, or a
// supervisor kill citing a closed reason token).
//
// THE FOLD. Fold(events, now, config) is a pure function from an append-only
// JSONL journal to the process table at one instant: per-proc state
// (RUNNING/DONE/KILLED), liveness class (LIVE/QUIET/STALLED), deadline
// overdue-ness, orphan-ness, and findings from a CLOSED verdict vocabulary —
// TOOL_DEADLINE_EXCEEDED (advice: kill), TOOL_HEARTBEAT_STALLED (advice:
// probe), TOOL_ORPHANED (advice: reap), TOOL_RESULT_AFTER_KILL (advice:
// quarantine_result). The reason codes live in the registered out-of-tree range
// (1040–1043, above abi.ReasonCoreMax) exactly like egressfloor's EGRESS_BLOCK;
// the consumer registers the names, so this leaf stays init-free and pure.
//
// WHAT THIS LEAF IS NOT (yet). It is the decision spine only — deterministic,
// offline-provable (`fak toolproc sample`), same-input ⇒ byte-identical-output.
// It does not itself kill processes, cancel MCP requests, or quarantine
// payloads; the enforcement wiring (gateway/guard supervisor emitting spawn and
// pulse events from the live wire and acting on the advice, a ResultAdmitter
// rung refusing post-kill payloads) is the labeled next step, kept honest by
// docs/notes/CONCEPT-TOOL-PROCESS-TABLE-2026-07-02.md.
//
// Tier: mechanism (2) — see internal/architest. This package may import only
// packages whose tier is <= 2 (it imports abi, tier 0); an upward import fails
// the architest gate. See AGENTS.md and internal/architest for the layering
// contract.
package toolproc
