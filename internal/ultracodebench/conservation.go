package ultracodebench

import (
	"encoding/json"
	"fmt"
	"sort"
)

// WorkClass gives every full-context token exactly one accounting owner.
type WorkClass string

const (
	WorkComputed   WorkClass = "computed"
	WorkScopedAway WorkClass = "scoped_away"
	WorkPrefixRead WorkClass = "prefix_read"
	WorkRepaired   WorkClass = "repaired"
	WorkUnknown    WorkClass = "unknown"
)

// OwnedSpan is a half-open token span in the full-context baseline.
type OwnedSpan struct {
	Start      int       `json:"start"`
	End        int       `json:"end"`
	Class      WorkClass `json:"class"`
	Provenance string    `json:"provenance"`
}

// ConservationReceipt preserves source telemetry while attaching normalized ownership.
type ConservationReceipt struct {
	Schema               string               `json:"schema"`
	Envelope             ScopedPrefixEnvelope `json:"envelope"`
	AcceptedOutcomeEqual bool                 `json:"accepted_outcome_equal"`
	BaselineTokens       int                  `json:"baseline_tokens"`
	RawProvider          json.RawMessage      `json:"raw_provider"`
	RawRuntime           json.RawMessage      `json:"raw_runtime"`
	Spans                []OwnedSpan          `json:"spans"`
}

// ConservationReport is the replayable conservation decision for one receipt.
type ConservationReport struct {
	Schema         string              `json:"schema"`
	Verdict        ScopedPrefixVerdict `json:"verdict"`
	Reason         string              `json:"reason"`
	BaselineTokens int                 `json:"baseline_tokens"`
	ByClass        map[WorkClass]int   `json:"by_class"`
	ClaimedSavings int                 `json:"claimed_savings"`
	Conserved      bool                `json:"conserved"`
}

// EvaluateConservation proves that normalized classes are disjoint and cover the
// full-context counterfactual. It never infers prefix reads from prompt deltas.
func EvaluateConservation(r ConservationReceipt) ConservationReport {
	out := ConservationReport{Schema: "fak.ultracode.conservation-report.v1", Verdict: ScopedPrefixAbstain, BaselineTokens: r.BaselineTokens, ByClass: map[WorkClass]int{}}
	if r.BaselineTokens <= 0 || len(missingEnvelope(r.Envelope)) > 0 || !json.Valid(r.RawProvider) || !json.Valid(r.RawRuntime) || string(r.RawProvider) == "null" || string(r.RawRuntime) == "null" {
		out.Reason = "missing full-context baseline or raw provider/runtime telemetry"
		return out
	}
	if !r.AcceptedOutcomeEqual {
		out.Reason = "accepted outcomes differ"
		return out
	}
	spans := append([]OwnedSpan(nil), r.Spans...)
	sort.Slice(spans, func(i, j int) bool { return spans[i].Start < spans[j].Start })
	cursor := 0
	valid := map[WorkClass]bool{WorkComputed: true, WorkScopedAway: true, WorkPrefixRead: true, WorkRepaired: true, WorkUnknown: true}
	for _, span := range spans {
		if !valid[span.Class] || span.Provenance == "" || span.Start != cursor || span.End <= span.Start || span.End > r.BaselineTokens {
			out.Reason = fmt.Sprintf("missing or overlapping ownership at token %d", cursor)
			return out
		}
		out.ByClass[span.Class] += span.End - span.Start
		cursor = span.End
	}
	if cursor != r.BaselineTokens {
		out.Reason = fmt.Sprintf("missing or overlapping ownership at token %d", cursor)
		return out
	}
	if out.ByClass[WorkUnknown] != 0 {
		out.Reason = "unknown span ownership"
		return out
	}
	out.ClaimedSavings = out.ByClass[WorkScopedAway] + out.ByClass[WorkPrefixRead]
	if out.ClaimedSavings > r.BaselineTokens {
		out.Reason = "claimed savings exceed full-context baseline"
		return out
	}
	out.Conserved = true
	out.Verdict = ScopedPrefixEnable
	out.Reason = "all full-context tokens have one authoritative accounting owner"
	return out
}
