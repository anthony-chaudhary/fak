package metrics

import (
	"errors"
	"fmt"
	"time"
)

// DwellRefusal is a stable, machine-readable reason an operator-dwell result
// could not be established.
type DwellRefusal string

const (
	DwellRefusalNone             DwellRefusal = ""
	DwellRefusalNoObservations   DwellRefusal = "no_observations"
	DwellRefusalInvalidTimestamp DwellRefusal = "invalid_timestamp"
	DwellRefusalNonMonotonic     DwellRefusal = "non_monotonic_observations"
)

// DwellObservation records the auditable inputs at one focus or visibility
// boundary. Visible and Focused are deliberately separate: neither implies the
// other.
type DwellObservation struct {
	At      time.Time `json:"at"`
	Visible bool      `json:"visible"`
	Focused bool      `json:"focused"`
}

// OperatorDwellResult describes observable focus time, never human attention.
// Complete is false while the final observation remains visible and focused,
// because that active interval has no witnessed endpoint and is not counted.
type OperatorDwellResult struct {
	Accepted             bool               `json:"accepted"`
	Refusal              DwellRefusal       `json:"refusal,omitempty"`
	ObservableFocusNanos int64              `json:"observable_focus_nanos"`
	Complete             bool               `json:"complete"`
	Observations         []DwellObservation `json:"observations"`
}

// MeasureOperatorDwell banks elapsed time only between adjacent observations
// whose starting state is both visible and focused. A blur and a visibility
// transition at the same instant therefore bank the interval exactly once.
func MeasureOperatorDwell(observations []DwellObservation) OperatorDwellResult {
	result := OperatorDwellResult{Observations: append([]DwellObservation(nil), observations...)}
	if len(observations) == 0 {
		result.Refusal = DwellRefusalNoObservations
		return result
	}
	for i, observation := range observations {
		if observation.At.IsZero() {
			result.Refusal = DwellRefusalInvalidTimestamp
			return result
		}
		if i > 0 && observation.At.Before(observations[i-1].At) {
			result.Refusal = DwellRefusalNonMonotonic
			return result
		}
		if i > 0 && observations[i-1].Visible && observations[i-1].Focused {
			result.ObservableFocusNanos += observation.At.Sub(observations[i-1].At).Nanoseconds()
		}
	}
	result.Accepted = true
	last := observations[len(observations)-1]
	result.Complete = !(last.Visible && last.Focused)
	return result
}

// ValidateOperatorDwellResult checks the public receipt's typed invariants.
func ValidateOperatorDwellResult(result OperatorDwellResult) error {
	measured := MeasureOperatorDwell(result.Observations)
	if measured.Accepted != result.Accepted || measured.Refusal != result.Refusal ||
		measured.ObservableFocusNanos != result.ObservableFocusNanos || measured.Complete != result.Complete {
		return errors.New("operator dwell receipt does not match its observations")
	}
	if result.ObservableFocusNanos < 0 {
		return fmt.Errorf("observable focus time cannot be negative")
	}
	return nil
}
