// Package toolprocgate is the revocation gate — the first ENFORCEMENT rung of
// the tool process table (seam 2 of
// docs/notes/CONCEPT-TOOL-PROCESS-TABLE-2026-07-02.md).
//
// internal/toolproc names the verdict: TOOL_RESULT_AFTER_KILL — a completion
// that arrives after the kernel revoked its call must not enter context as if
// the call were live. This leaf makes that verdict bite. It keeps an
// in-process revocation table (Kill / KilledReason: ToolCall.TraceID → the
// closed reason token the kill cited) and registers a rank-2
// abi.ResultAdmitter that quarantines any result whose call id is in the
// table: the payload is stubbed IN PLACE before the content screens
// (secretgate 4, normgate 5, ctxmmu 10, ifc 20) ever see the raw bytes, and
// the verdict cites toolproc.ReasonToolResultAfterKill from the registered
// out-of-tree vocabulary.
//
// DROPPED, NOT HELD. A quarantined secret is CAS-pinned for a witnessed
// page-in; a post-kill payload has no legitimate re-entry path, so this gate
// drops the original bytes — fail-closed here means fail-forgotten. Only the
// byte length survives, on the stub, for forensics.
//
// REGISTERED-BUT-INERT BY DEFAULT. With an empty revocation table the gate
// Defers on every result, so the defconfig ships it enabled at zero behavior
// change. Teeth engage the moment something calls Kill.
//
// THE SUPERVISOR (seam 1's engine, supervisor.go). NewSupervisor gives an
// embedder — the gateway proxy, `fak guard`, the agent loop, the MCP server —
// the live side of the table: report Spawn (with the in-flight work's cancel
// lever) / Pulse / Exit / SessionEnd as they cross the wire, call Tick on
// your cadence. Tick folds the same pure toolproc.Fold the CLI uses, then
// ACTS: kill/reap advice cancels the in-flight work once, enters the call
// into the revocation table (arming the Gate against its late completion),
// and appends the kill to the journal; probe stays advisory. Clock-free and
// goroutine-free — the embedder owns the cadence, tests own the clock. What
// remains for the wire adapters (gateway/guard/MCP) is only observation
// plumbing: map their dispatch/stream/poll events onto these entry points.
// Cross-process revocation (a CLI killing a call inside a running gateway)
// rides on that adapter, not this leaf.
//
// Tier: integrator (4) — see internal/architest. This package may import only
// packages whose tier is <= 4 (it imports abi tier 0 and toolproc tier 2); an
// upward import fails the architest gate. See AGENTS.md and
// internal/architest for the layering contract.
package toolprocgate
