package model

import (
	"errors"
	"testing"
)

// Failure formatting must not make the caller's expected-shape array escape
// during the live per-token metadata scan.
// fak-test:runtime fast est=10ms lane=default
func TestV41AdmissionShapeSuccessAllocations(t *testing.T) {
	const name = "model.layers.0.ffn.experts.0.w1.weight"
	m := &Model{manifest: map[string]tensorMeta{name: {Shape: []int{2304, 5120}}}}
	allocs := testing.AllocsPerRun(100, func() {
		if err := m.v41AdmitShape(name, v41StageMoE, 0, 2304, 5120); err != nil {
			panic(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("successful matrix admission allocated %g objects, want zero", allocs)
	}
	// Admission still reads mutable metadata on every invocation.
	m.manifest[name] = tensorMeta{Shape: []int{2305, 5120}}
	err := m.v41AdmitShape(name, v41StageMoE, 0, 2304, 5120)
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("mutated shape was not refused: %v", err)
	}
	delete(m.manifest, name)
	if err := m.v41AdmitShape(name, v41StageMoE, 0, 2304, 5120); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("removed shape was not refused: %v", err)
	}
}
