//go:build !darwin || !arm64 || !cgo

package metalgemm

import "testing"

func TestExecutionObservationUnavailableIsTyped(t *testing.T) {
	observation := NewExecutionObservation(ExecutionQ4KGEMV)
	_, err := observation.Snapshot()
	if !IsExecutionEventsUnavailable(err) {
		t.Fatalf("Snapshot error = %v, want typed unavailable", err)
	}
}
