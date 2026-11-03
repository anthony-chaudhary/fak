// events.go — the closed core EventKind vocabulary (additive), keyed off by KPI taps,
// the vDSO cache-fill, stewards, and the self-labeling rung harvester; numbers come
// from the reserved EventsCore / EventsLabel blocks in registry.go.
package abi

const (
	// EvSubmit marks a call entering the kernel; the KPI taps and stewards key off this entry point.
	EvSubmit EventKind = iota
	// EvDecide fires when the adjudicator chain resolves a verdict; observers fold the stream and skip redundant re-carries.
	EvDecide
	// EvDeny pairs with a denying EvDecide: the call was refused and the verdict carries the ReasonCode.
	EvDeny
	// EvDispatch marks an allowed call handed to the engine for execution.
	EvDispatch
	// EvComplete marks a dispatched call finished by the engine; carries the result and engine cost, and the vDSO fills its cache from it.
	EvComplete
	// EvQuarantine marks a result held out of context by the MMU instead of admitting it to the transcript.
	EvQuarantine
	// EvVDSOHit marks a call served locally by the vDSO cache instead of reaching the provider; the IFC ledger raises from it.
	EvVDSOHit
	// EvResultDeny marks a produced result hard-refused by the result-admit stack, after the call already ran.
	EvResultDeny
	// EvCapFault marks a capability faulted in from cold storage, the page-miss half of the capability pager.
	EvCapFault
	// EvCapEvict marks a capability paged out of hot storage, the eviction twin of EvCapFault.
	EvCapEvict
	// EvCapVersionBind marks a capability version bound to the table, a page-table remap.
	EvCapVersionBind
	// EvRungLabel lives in the EventsLabel block (>=128): a typed LabelRow rode the event.
	EvRungLabel EventKind = 128
)

// RedundantDecisionEvent reports whether ev is a SECOND emission of a decision another
// event already carries, so an observer folding the decision stream into one record per
// decided call skips it to avoid double-counting. The kernel emits more than one event
// per logical decision: every verdict fires EvDecide, a deny ALSO fires a paired EvDeny,
// and the require-witness Submit path fires an intermediate EvDecide before the resolved
// decision. An EvDecide is redundant when re-carried by its paired EvDeny (Deny) or
// followed by the final allow/deny of a SUBMITTED require-witness call (SeqNo != 0); a
// pure Decide() call (SeqNo == 0) has no follow-up, so its verdict is final, NOT redundant.
func RedundantDecisionEvent(ev Event) bool {
	if ev.Kind != EvDecide || ev.Verdict == nil {
		return false
	}
	switch ev.Verdict.Kind {
	case VerdictDeny:
		return true
	case VerdictRequireWitness:
		return ev.Call != nil && ev.Call.SeqNo != 0
	default:
		return false
	}
}
