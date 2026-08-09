package guardaccuracy

import (
	"encoding/json"
	"fmt"
	"io"
)

// AdmittedCall is a successful tool call that the guard admitted. Replaying
// replay rules over this corpus measures known false rejections: every
// replayed rejection would have blocked a call whose outcome is known-good.
type AdmittedCall struct {
	Tool       string          `json:"tool"`
	Arguments  json.RawMessage `json:"arguments,omitempty"`
	Admitted   bool            `json:"admitted"`
	Successful bool            `json:"successful"`
}

// ReplayRule names a proposed rejection rule and its tolerated number
// of false rejections over an admitted-successful replay corpus.
type ReplayRule struct {
	Name    string
	Ceiling int
	Reject  func(AdmittedCall) bool
}

// PredicateReplay reports the measured false-rejection count for one candidate.
type PredicateReplay struct {
	Name            string `json:"name"`
	CallsEvaluated  int    `json:"calls_evaluated"`
	FalseRejections int    `json:"false_rejections"`
	Ceiling         int    `json:"ceiling"`
}

// LoadAdmittedCalls decodes a fixture journal and refuses entries that are not
// admitted and successful. That keeps the replay denominator explicit rather
// than silently treating unknown or failed calls as false rejections.
func LoadAdmittedCalls(r io.Reader) ([]AdmittedCall, error) {
	var calls []AdmittedCall
	if err := json.NewDecoder(r).Decode(&calls); err != nil {
		return nil, fmt.Errorf("decode admitted-call journal: %w", err)
	}
	for i, call := range calls {
		if call.Tool == "" {
			return nil, fmt.Errorf("admitted-call journal entry %d: tool is required", i)
		}
		if !call.Admitted || !call.Successful {
			return nil, fmt.Errorf("admitted-call journal entry %d: corpus requires admitted, successful calls", i)
		}
	}
	return calls, nil
}

// ReplayPredicates evaluates candidates against known-good calls. It returns
// every measured report even when a ceiling is exceeded, so callers can show
// the evidence that caused the gate to fail.
func ReplayPredicates(calls []AdmittedCall, candidates ...ReplayRule) ([]PredicateReplay, error) {
	reports := make([]PredicateReplay, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate.Name == "" || candidate.Reject == nil {
			return nil, fmt.Errorf("replay rule requires a name and reject function")
		}
		if candidate.Ceiling < 0 {
			return nil, fmt.Errorf("replay rule %q: ceiling must be non-negative", candidate.Name)
		}
		report := PredicateReplay{Name: candidate.Name, CallsEvaluated: len(calls), Ceiling: candidate.Ceiling}
		for _, call := range calls {
			if candidate.Reject(call) {
				report.FalseRejections++
			}
		}
		reports = append(reports, report)
	}
	for _, report := range reports {
		if report.FalseRejections > report.Ceiling {
			return reports, fmt.Errorf("replay rule %q: false rejections %d exceed ceiling %d", report.Name, report.FalseRejections, report.Ceiling)
		}
	}
	return reports, nil
}
