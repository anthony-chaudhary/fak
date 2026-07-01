// Rung F2 (issue #1885): the externalize gate. F1 (#1884) DETECTS transcript-only
// load-bearing state; this rung REFUSES to rotate while any exists, emitting the closed
// RELAY_NOT_EXTERNALIZED reason so the leg externalizes the state (into a durable pointer)
// or parks — instead of rotating and silently dropping it. Fail-closed: a non-empty
// detector blocks the rotate. No auto-remediation (the leg must act on the refusal).
package relay

// ReasonNotExternalized is the relay reason token emitted when a rotation is refused
// because load-bearing state is still transcript-only (RELAY-REASON-VOCABULARY-2026-07-01.md).
const ReasonNotExternalized = "RELAY_NOT_EXTERNALIZED"

// ExternalizeGate is the decision of the pre-rotate externalize check: whether the rotate
// is admitted, and — when refused — the reason token plus the transcript-only Culprits the
// leg must externalize or park on. When Admit is true the other fields are zero.
type ExternalizeGate struct {
	Admit    bool              `json:"admit"`
	Reason   string            `json:"reason,omitempty"`
	Culprits []LoadBearingFact `json:"culprits,omitempty"`
}

// CheckExternalizeGate refuses rotation with RELAY_NOT_EXTERNALIZED when any load-bearing
// fact is still transcript-only (the F1 detector is non-empty); otherwise it admits the
// rotate. Pure over its input — the fail-closed admission kernel for the safe point's
// externalize precondition.
func CheckExternalizeGate(facts []LoadBearingFact) ExternalizeGate {
	unbacked := TranscriptOnly(facts)
	if len(unbacked) == 0 {
		return ExternalizeGate{Admit: true}
	}
	return ExternalizeGate{Admit: false, Reason: ReasonNotExternalized, Culprits: unbacked}
}
