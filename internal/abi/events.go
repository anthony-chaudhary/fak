// events.go — the closed core EventKind vocabulary (additive). Emitters observe
// these lifecycle transitions; KPI taps, the vDSO cache-fill, stewards, and the
// self-labeling rung harvester all key off them. Numbers are drawn from the
// reserved EventsCore / EventsLabel blocks in registry.go.
package abi

const (
	EvSubmit     EventKind = iota // a call entered the kernel
	EvDecide                      // the adjudicator chain resolved a verdict
	EvDeny                        // a call was refused (verdict carries the reason)
	EvDispatch                    // an allowed call was dispatched to the engine
	EvComplete                    // an engine produced a result (Result is set)
	EvQuarantine                  // a result was held out of context by the MMU
	EvVDSOHit                     // a call was served locally by the vDSO
	EvResultDeny                  // a produced result was hard-refused by the result-admit stack
	// EvRungLabel lives in the EventsLabel block (>=128): a typed LabelRow rode the
	// event (the pre-flight self-labeling signal).
	EvRungLabel EventKind = 128
)

// RedundantDecisionEvent reports whether ev is a SECOND emission of a decision
// that another event already carries — so an observer that folds the decision
// stream into ONE record per decided call must skip it to avoid double-counting.
//
// The kernel emits more than one event per logical decision (see Kernel.Decide /
// Kernel.Submit): EvDecide fires for EVERY resolved verdict, and a deny ALSO fires
// a dedicated EvDeny carrying the same verdict; the require-witness Submit path
// fires an intermediate EvDecide (RequireWitness) and THEN the resolved decision
// (an EvDecide allow, or an EvDeny). The EvDecide that another event re-carries is
// the one for:
//   - a Deny verdict — the paired EvDeny is always emitted right after it, and
//   - a RequireWitness verdict on a SUBMITTED call (Call.SeqNo != 0) — the kernel
//     resolves the gate and emits the final allow/deny next. A pure Decide() call
//     has SeqNo == 0 and no follow-up, so ITS require-witness verdict is final and
//     NOT redundant.
//
// This is the single definition of the de-double rule the decision-stream folders
// share (internal/journal, internal/harvest, internal/trajectory);
// internal/rungobs is the reference consumer, which additionally keeps a per-SeqNo
// seen-set for belt-and-braces. A consumer that subscribes to ONE event kind (the
// adjudicator promote ledger, rulesynth) or to EvSubmit (tracesink) cannot
// double-count and does not need this.
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
