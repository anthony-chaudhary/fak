package gateway

// a2a_progress.go — carry fak's no-`claimed`-field status invariant across the A2A
// HTTP edge. The in-kernel status discipline (dos_status, internal/relay/progress.go's
// VerifiedProgress) guarantees a peer is never handed a self-report: progress is a
// RE-VERIFIABLE cursor into the intent ledger, never a number a closing leg asserted.
// The A2A task edge (a2a.go) is the one surface a FOREIGN in-flight agent actually
// reads to understand ours — and until now GetTask returned only the imperative
// State/Result (handleA2ASendMessage even hardcodes a "completed"/success self-report).
// This projects the SAME VerifiedProgress shape onto that read, so a peer doing
// GET /a2a/v1/tasks/{id} receives an evidence-shaped progress pointer alongside the
// state. Reuse, not a new format: it reads through internal/relay's cursor.
//
// Fail-closed by construction: a task with no bound intent-ledger anchor
// (LedgerRef == "") yields verdict "unknown" with no steps — the peer is handed
// nothing it cannot re-verify, never an implied "verified/completed". Sitting next to
// a simulated state:"completed", a progress verdict of "unknown" is exactly the honest
// signal that the completion is NOT ledger-verified. Wiring a live file-backed
// relay.LedgerReader onto a task's run is the named next rung (relay/progress.go says
// the same for its in-tree twin); this rung carries the invariant to the edge.

import "github.com/anthony-chaudhary/fak/internal/relay"

// a2aVerifiedProgress projects the no-`claimed`-field verified-progress cursor onto an
// A2A task. It never returns a self-report: the result is always relay.VerifiedProgress
// (verdict + ledger-sourced steps + reason, no `claimed`/`success` field anywhere in the
// type tree — relay.TestVerifiedProgressHasNoClaimedField pins that reflectively).
//
// Fail-closed on every unbound or unreadable edge:
//   - nil task            -> unknown (nothing to read)
//   - empty LedgerRef      -> unknown (relay.ReadVerifiedProgress returns before it
//     touches lr, so a nil reader is safe here)
//   - LedgerRef set, lr nil -> unknown (a named anchor with no reader wired must not be
//     reported as verified; guard before relay would dereference lr)
//
// so a peer only ever receives progress it can re-verify.
func a2aVerifiedProgress(task *a2aTask, lr relay.LedgerReader) relay.VerifiedProgress {
	if task == nil {
		return relay.VerifiedProgress{
			Verdict: relay.ProgressUnknown,
			Reason:  "no task to read verified progress for; failing closed",
		}
	}
	if task.LedgerRef != "" && lr == nil {
		return relay.VerifiedProgress{
			Verdict:   relay.ProgressUnknown,
			LedgerRef: task.LedgerRef,
			Reason:    "task names an intent-ledger anchor but no LedgerReader is wired; failing closed rather than asserting progress",
		}
	}
	return relay.ReadVerifiedProgress(relay.ProgressCursor{LedgerRef: task.LedgerRef}, lr)
}
