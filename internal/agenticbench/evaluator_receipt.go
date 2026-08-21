package agenticbench

import (
	"fmt"
	"math"
)

const (
	EvaluatorSuccess = "success"
	EvaluatorFailure = "failure"
	EvaluatorTimeout = "timeout"
)

type EvaluatorReceipt struct {
	Status     string
	Score      *float64
	MinScore   float64
	MaxScore   float64
	DurationMS int64
	ErrorRef   string
}

func (r EvaluatorReceipt) Validate() error {
	if r.DurationMS < 0 {
		return fmt.Errorf("evaluator duration must be non-negative")
	}
	switch r.Status {
	case EvaluatorSuccess:
		if r.Score == nil {
			return fmt.Errorf("successful evaluator receipt requires score")
		}
		if math.IsNaN(*r.Score) || math.IsInf(*r.Score, 0) {
			return fmt.Errorf("evaluator score must be finite")
		}
		if r.MinScore > r.MaxScore || *r.Score < r.MinScore || *r.Score > r.MaxScore {
			return fmt.Errorf("evaluator score outside declared domain")
		}
		if r.ErrorRef != "" {
			return fmt.Errorf("successful evaluator receipt cannot carry error evidence")
		}
	case EvaluatorFailure, EvaluatorTimeout:
		if r.Score != nil {
			return fmt.Errorf("non-success evaluator receipt cannot carry score")
		}
		if r.ErrorRef == "" {
			return fmt.Errorf("non-success evaluator receipt requires error evidence")
		}
	default:
		return fmt.Errorf("unknown evaluator status %q", r.Status)
	}
	return nil
}
