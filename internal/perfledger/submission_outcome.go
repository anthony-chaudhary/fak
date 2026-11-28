// Structural adaptation of ai-dynamo/aiperf submission-validity fold with reason
// codes (src/aiperf/exporters/aggregate/aggregate_base_exporter.py:69 @ ab610add4e88,
// Apache-2.0); clean-room Go.
package perfledger

import "sort"

// SubmissionReason is a closed reason code explaining why a benchmark submission
// is not valid.
type SubmissionReason string

const (
	ReasonFailedRequests  SubmissionReason = "failed_requests"
	ReasonZeroDuration    SubmissionReason = "zero_duration"
	ReasonNoRequests      SubmissionReason = "no_requests"
	ReasonMissingEnvelope SubmissionReason = "missing_envelope"
	ReasonGateFailed      SubmissionReason = "faithfulness_gate_failed"
)

// SubmissionInput carries the facts folded into a validity verdict.
type SubmissionInput struct {
	Requests        int
	FailedRequests  int
	DurationSeconds float64
	HasEnvelope     bool
	GatePassed      bool
}

// SubmissionOutcome is the folded verdict: Valid is true only when no reason holds.
type SubmissionOutcome struct {
	Valid   bool               `json:"valid"`
	Reasons []SubmissionReason `json:"reasons,omitempty"`
}

// FoldSubmissionOutcome folds every failure fact into sorted, de-duplicated
// reason codes. Fail-closed: any reason invalidates the whole submission.
func FoldSubmissionOutcome(in SubmissionInput) SubmissionOutcome {
	set := map[SubmissionReason]struct{}{}
	if in.Requests <= 0 {
		set[ReasonNoRequests] = struct{}{}
	}
	if in.FailedRequests > 0 {
		set[ReasonFailedRequests] = struct{}{}
	}
	if !(in.DurationSeconds > 0) {
		set[ReasonZeroDuration] = struct{}{}
	}
	if !in.HasEnvelope {
		set[ReasonMissingEnvelope] = struct{}{}
	}
	if !in.GatePassed {
		set[ReasonGateFailed] = struct{}{}
	}
	if len(set) == 0 {
		return SubmissionOutcome{Valid: true}
	}
	reasons := make([]SubmissionReason, 0, len(set))
	for r := range set {
		reasons = append(reasons, r)
	}
	sort.Slice(reasons, func(i, j int) bool { return reasons[i] < reasons[j] })
	return SubmissionOutcome{Valid: false, Reasons: reasons}
}
