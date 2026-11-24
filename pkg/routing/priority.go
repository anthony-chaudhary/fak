package routing

import "strings"

// PriorityClass is a tenant-asserted commercial priority tier. It is a closed
// vocabulary: premium and flex, plus the absence of a class. The fabric
// reorders a resolved chain for it under a peak signal; it never prices,
// admits, or refuses — the pricing coupling and the admission wiring are
// later phases that consume this seam.
type PriorityClass string

const (
	// PriorityClassNone: no class asserted. The chain keeps its declared
	// order.
	PriorityClassNone PriorityClass = ""
	// PriorityClassPremium: the protected tier. Under an active peak signal,
	// a premium request puts flex members at the back of its chain.
	PriorityClassPremium PriorityClass = "premium"
	// PriorityClassFlex: the delay-tolerant tier. Its requests keep the
	// declared order; de-prioritization is the peak reordering's effect on
	// the shared chain, not a separate walk.
	PriorityClassFlex PriorityClass = "flex"
)

// PriorityClasses returns the closed class vocabulary in declaration order.
// The slice is a fresh copy, so a caller cannot reorder the vocabulary out
// from under the next one.
func PriorityClasses() []string {
	return []string{
		string(PriorityClassNone),
		string(PriorityClassPremium),
		string(PriorityClassFlex),
	}
}

// ParsePriorityClass maps a tenant-asserted value to a closed PriorityClass;
// anything unrecognised is PriorityClassNone.
func ParsePriorityClass(v string) PriorityClass {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case string(PriorityClassPremium):
		return PriorityClassPremium
	case string(PriorityClassFlex):
		return PriorityClassFlex
	default:
		return PriorityClassNone
	}
}

// PeakSignal is the caller's peak-condition verdict, carried as data. The
// fabric never measures load, clocks, or queues; the caller (admission, a
// governor, an operator) decides and says so here.
type PeakSignal bool

// PriorityMember is one resolved chain entry as the priority order sees it.
type PriorityMember struct {
	// Name is the member's identity in the chain.
	Name string
	// Class is the member's declared priority class. None means the member is
	// not classed and keeps its declared position.
	Class PriorityClass
}

// OrderForPriority returns the chain order a priority class asks for and
// whether the class took effect. Under an active peak signal, a premium
// request puts flex members at the back: every non-flex member in declared
// order, then every flex member in declared order. A flex request, a
// classless request, or any request outside peak keeps the declared order
// unchanged. It takes effect only when at least one flex member is present;
// otherwise the declared order is returned unchanged.
//
// The function is pure and replayable: the same inputs always yield a
// byte-identical order, and the input slice is never mutated.
func OrderForPriority(members []PriorityMember, class PriorityClass, peak PeakSignal) ([]string, bool) {
	declared := make([]string, len(members))
	for i, m := range members {
		declared[i] = m.Name
	}
	if !peak || class != PriorityClassPremium {
		return declared, false
	}
	var head, tail []string
	for _, m := range members {
		if m.Class == PriorityClassFlex {
			tail = append(tail, m.Name)
		} else {
			head = append(head, m.Name)
		}
	}
	if len(tail) == 0 {
		return declared, false
	}
	out := make([]string, 0, len(members))
	out = append(out, head...)
	return append(out, tail...), true
}
