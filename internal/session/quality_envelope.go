package session

import "sort"

// quality_envelope.go defines the SESSION-START quality envelope (#1964, QA-dogfood
// spine #1961/QD-004): the single origin record saying which QA controls govern a
// session. Session budget envelopes already exist (budget_envelope.go), but a new
// session carried no one record answering "which witness policy, which dogfood probes,
// which scorecards apply to THIS run?" — that context lived scattered across config and
// was re-derived (or forgotten) on inspect. This is that record: deterministic data
// stamped once at session start, persisted verbatim into the portable image
// (internal/sessionimage) so a restored/inspected session exposes it unchanged rather
// than re-deriving it.

// QualityEnvelope is the origin record of the QA controls that govern a session: the
// budget axes it opened under, the witness policy that gates its claims, the dogfood
// probes expected to run at origin, and the control-pane scorecards it is a member of.
// It is deterministic data only — a value a session snapshot carries, not runtime
// state. Canonical makes its byte form stable so it rides the image's sha256 integrity
// index like every other part.
type QualityEnvelope struct {
	// Budget is the budget envelope the session opened under (the same axes
	// budget_envelope.go parses), recorded so the origin record is self-contained.
	Budget BudgetEnvelope `json:"budget"`
	// WitnessPolicy names the evidence class the session's claims must satisfy — e.g.
	// "proof-by-default" (a captured artifact) or "dos-verify" (a plan/phase referee).
	WitnessPolicy string `json:"witness_policy,omitempty"`
	// DogfoodProbes are the at-origin scorecard/check probes expected to run for this
	// session (the QA-dogfood spine's "run the score where the work is created").
	DogfoodProbes []string `json:"dogfood_probes,omitempty"`
	// ScorecardCards are the control-pane scorecards this session is a member of (their
	// stable card keys, e.g. "code_quality", "milestone_scorecard").
	ScorecardCards []string `json:"scorecard_cards,omitempty"`
}

// IsZero reports the permissive default: no witness policy, no probes, no scorecard
// membership, and a zero budget envelope. Supports treating an absent envelope as "no
// QA controls declared" without a nil pointer.
func (e QualityEnvelope) IsZero() bool {
	return e.WitnessPolicy == "" && len(e.DogfoodProbes) == 0 && len(e.ScorecardCards) == 0 &&
		e.Budget == (BudgetEnvelope{})
}

// Canonical returns a copy with the probe and scorecard lists deduplicated and sorted,
// so a fixed set of controls serializes to byte-identical bytes regardless of the order
// they were declared in — the determinism the image's integrity index and .faksession
// archive rely on. It never mutates the receiver's slices.
func (e QualityEnvelope) Canonical() QualityEnvelope {
	out := e
	out.DogfoodProbes = dedupeSorted(e.DogfoodProbes)
	out.ScorecardCards = dedupeSorted(e.ScorecardCards)
	return out
}

// dedupeSorted returns a new sorted slice with empty and duplicate entries removed, or
// nil when nothing survives (so an all-empty input canonicalizes to the zero value, not
// an empty non-nil slice).
func dedupeSorted(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		if s == "" {
			continue
		}
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	if len(out) == 0 {
		return nil
	}
	sort.Strings(out)
	return out
}
