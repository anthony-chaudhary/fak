package superstream

import (
	"github.com/anthony-chaudhary/fak/internal/laneadmit"
)

// Stepper amortizes queue normalization across a whole stream drain.
//
// DecideStep normalizes the StreamSpec on every call, so a decision over a
// Q-item queue copies the queue once and draining Q items allocates
// Theta(Q^2) bytes in total. An executor that will step more than once
// should construct a Stepper once and call Step for every decision: the
// normalized spec is held for the lifetime of the drain, so per-step cost
// is O(1) in queue length and the drain allocates Theta(Q) total.
//
// Stepper is not concurrency-safe; each executing goroutine owns one.
// ApplyStep remains a free function: state transition mutates the caller's
// StreamState in place and never touches the spec.
type Stepper struct {
	norm   StreamSpec
	holder string
}

// NewStepper normalizes the spec once and returns a stepper that serves the
// whole drain from that single copy.
func NewStepper(spec StreamSpec, holder string) *Stepper {
	return &Stepper{norm: spec.NormalizedSpec(), holder: holder}
}

// Step computes the single next action to progress the workstream, using the
// spec copy captured at construction. Semantics are identical to DecideStep
// over the same normalized spec: same action, same reason strings, same
// safety verdict for the same state, leases, and taxonomy.
func (s *Stepper) Step(state StreamState, liveLeases []laneadmit.Lease, tax laneadmit.Taxonomy) StepDecision {
	return decideStepNormalized(s.norm, state, s.holder, liveLeases, tax)
}

// Norm returns the normalized spec the stepper serves decisions from. The
// caller must treat the returned Queue slice as read-only (it is the one
// normalized copy backing the whole drain).
func (s *Stepper) Norm() StreamSpec { return s.norm }
