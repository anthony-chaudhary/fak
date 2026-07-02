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
// change. Teeth engage the moment a supervisor calls Kill — the gateway/guard
// tick acting on a toolproc `kill` advice (seam 1, next step), or any
// in-process caller. Cross-process revocation (a CLI killing a call inside a
// running gateway) rides on seam 1's supervisor, not this leaf.
//
// Tier: integrator (4) — see internal/architest. This package may import only
// packages whose tier is <= 4 (it imports abi tier 0 and toolproc tier 2); an
// upward import fails the architest gate. See AGENTS.md and
// internal/architest for the layering contract.
package toolprocgate
