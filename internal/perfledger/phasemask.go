// Adapted from ai-dynamo/aiperf
// modified from the original): phase-scoped masking by categorical tag rather
// than wall-clock windowing.
package perfledger

import (
	"errors"
	"math"
)

// PhaseSummary aggregates a numeric metric over records selected strictly by a
// categorical phase tag.
type PhaseSummary struct {
	PhaseTag   string
	PhaseValue string
	Matched    int
	Count      int
	Sum        float64
	Mean       float64
}

// PhaseMask returns a boolean mask selecting records whose categorical tag
// matches phaseValue, ignoring wall-clock timestamps.
func PhaseMask(s *ColumnStore, phaseTag, phaseValue string) ([]bool, error) {
	if s == nil {
		return nil, errors.New("perfledger: nil ColumnStore")
	}
	if phaseTag == "" || phaseValue == "" {
		return nil, errors.New("perfledger: phaseTag and phaseValue are required")
	}
	return s.MaskForCategorical(phaseTag, phaseValue), nil
}

// PhaseScopedNumeric extracts the non-NaN values of metricTag for records
// matching the categorical phase tag/value.
func PhaseScopedNumeric(s *ColumnStore, metricTag, phaseTag, phaseValue string) ([]float64, PhaseSummary, error) {
	mask, err := PhaseMask(s, phaseTag, phaseValue)
	if err != nil {
		return nil, PhaseSummary{}, err
	}
	vals := s.Numeric(metricTag)
	summary := PhaseSummary{
		PhaseTag:   phaseTag,
		PhaseValue: phaseValue,
	}
	var out []float64
	for i, hit := range mask {
		if !hit {
			continue
		}
		summary.Matched++
		if i < len(vals) && !math.IsNaN(vals[i]) {
			out = append(out, vals[i])
			summary.Count++
			summary.Sum += vals[i]
		}
	}
	if summary.Count > 0 {
		summary.Mean = summary.Sum / float64(summary.Count)
	}
	return out, summary, nil
}
