package compute

import "testing"

func TestGraphArgumentPromotionMatchesIndependentCPUOracle(t *testing.T) {
	base := GraphArgumentPromotionCandidate{
		Name:            "state",
		Direction:       GraphArgumentInputOutput,
		SizeBytes:       16,
		MaxSizeBytes:    16,
		ByReference:     true,
		DirectCallUses:  true,
		CompleteCallSCC: true,
	}

	tests := []struct {
		name      string
		candidate GraphArgumentPromotionCandidate
	}{
		{name: "mutable input and output", candidate: base},
		{name: "input", candidate: alterGraphArgumentCandidate(base, func(c *GraphArgumentPromotionCandidate) { c.Direction = GraphArgumentInput })},
		{name: "output", candidate: alterGraphArgumentCandidate(base, func(c *GraphArgumentPromotionCandidate) { c.Direction = GraphArgumentOutput })},
		{name: "boundary", candidate: alterGraphArgumentCandidate(base, func(c *GraphArgumentPromotionCandidate) { c.SizeBytes = c.MaxSizeBytes })},
		{name: "oversized", candidate: alterGraphArgumentCandidate(base, func(c *GraphArgumentPromotionCandidate) { c.SizeBytes++ })},
		{name: "exported", candidate: alterGraphArgumentCandidate(base, func(c *GraphArgumentPromotionCandidate) { c.Exported = true })},
		{name: "indirect use", candidate: alterGraphArgumentCandidate(base, func(c *GraphArgumentPromotionCandidate) { c.DirectCallUses = false })},
		{name: "incomplete recursive component", candidate: alterGraphArgumentCandidate(base, func(c *GraphArgumentPromotionCandidate) { c.CompleteCallSCC = false })},
		{name: "volatile", candidate: alterGraphArgumentCandidate(base, func(c *GraphArgumentPromotionCandidate) { c.VolatileAccess = true })},
		{name: "captured", candidate: alterGraphArgumentCandidate(base, func(c *GraphArgumentPromotionCandidate) { c.Captured = true })},
		{name: "escaping", candidate: alterGraphArgumentCandidate(base, func(c *GraphArgumentPromotionCandidate) { c.Escapes = true })},
		{name: "unsafe projection", candidate: alterGraphArgumentCandidate(base, func(c *GraphArgumentPromotionCandidate) { c.UnsafeProjection = true })},
		{name: "not by reference", candidate: alterGraphArgumentCandidate(base, func(c *GraphArgumentPromotionCandidate) { c.ByReference = false })},
		{name: "unknown direction", candidate: alterGraphArgumentCandidate(base, func(c *GraphArgumentPromotionCandidate) { c.Direction = "unknown" })},
		{name: "invalid size", candidate: alterGraphArgumentCandidate(base, func(c *GraphArgumentPromotionCandidate) { c.SizeBytes = 0 })},
		{name: "invalid limit", candidate: alterGraphArgumentCandidate(base, func(c *GraphArgumentPromotionCandidate) { c.MaxSizeBytes = 0 })},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SelectGraphArgumentPromotion(tt.candidate)
			want := cpuReferenceGraphArgumentPromotion(tt.candidate)
			if got != want {
				t.Fatalf("selection = %+v, CPU reference oracle = %+v", got, want)
			}
			if !got.Eligible && got.Action != GraphArgumentKeepByReference {
				t.Fatalf("ineligible selection did not retain current ABI: %+v", got)
			}
		})
	}
}

func alterGraphArgumentCandidate(candidate GraphArgumentPromotionCandidate, alter func(*GraphArgumentPromotionCandidate)) GraphArgumentPromotionCandidate {
	alter(&candidate)
	return candidate
}

// cpuReferenceGraphArgumentPromotion deliberately does not call production
// selector helpers. It models the old CPU ABI first, then applies the reference
// caller/callee rewrite only when every safety fact is present.
func cpuReferenceGraphArgumentPromotion(candidate GraphArgumentPromotionCandidate) GraphArgumentPromotionSelection {
	selection := GraphArgumentPromotionSelection{
		Name:   candidate.Name,
		Action: GraphArgumentKeepByReference,
	}

	var rejected string
	if !candidate.ByReference {
		rejected = "not-by-reference"
	} else if candidate.Exported {
		rejected = "exported"
	} else if !candidate.DirectCallUses {
		rejected = "non-direct-call-use"
	} else if !candidate.CompleteCallSCC {
		rejected = "incomplete-call-scc"
	} else if candidate.VolatileAccess {
		rejected = "volatile-access"
	} else if candidate.Captured {
		rejected = "captured"
	} else if candidate.Escapes {
		rejected = "escapes"
	} else if candidate.UnsafeProjection {
		rejected = "unsafe-projection"
	} else if candidate.SizeBytes < 1 {
		rejected = "invalid-size"
	} else if candidate.MaxSizeBytes < 1 {
		rejected = "invalid-size-limit"
	} else if candidate.SizeBytes > candidate.MaxSizeBytes {
		rejected = "oversized"
	} else if candidate.Direction != GraphArgumentInput && candidate.Direction != GraphArgumentOutput && candidate.Direction != GraphArgumentInputOutput {
		rejected = "unsupported-direction"
	}
	if rejected != "" {
		selection.Reason = rejected
		return selection
	}

	selection.Eligible = true
	if candidate.Direction == GraphArgumentInput {
		selection.Action = GraphArgumentPromoteToValue
	} else if candidate.Direction == GraphArgumentOutput {
		selection.Action = GraphArgumentPromoteToResult
	} else {
		selection.Action = GraphArgumentPromoteToBoth
	}
	return selection
}
