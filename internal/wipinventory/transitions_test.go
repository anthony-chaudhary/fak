package wipinventory

import (
	"strings"
	"testing"
	"time"
)

var testTime = time.Date(2026, 9, 1, 8, 0, 0, 0, time.UTC)

func id(n byte) WIPUnitID {
	const digits = "0123456789abcdef"
	b := []byte("wip:v1:00000000000000000000000000000000")
	b[len(b)-2] = digits[n>>4]
	b[len(b)-1] = digits[n&15]
	return WIPUnitID(b)
}

func tr(kind TransitionKind, predecessors, successors []WIPUnitID) Transition {
	return Transition{Kind: kind, Timestamp: testTime, Source: "test", Provenance: Provenance{Actor: "tester", Mechanism: "fixture"}, Predecessors: predecessors, Successors: successors, Witness: "witness"}
}

func TestEveryLegalTransition(t *testing.T) {
	one, two, three, four, five := id(1), id(2), id(3), id(4), id(5)
	bind := tr(TransitionBind, []WIPUnitID{one}, []WIPUnitID{one})
	bind.References = []SurfaceReference{{Kind: SurfaceLaneLease, LaneLease: &LaneLeaseReference{Lane: "wipinventory", LeaseID: "lease-1"}}}
	history := History{Schema: WIPUnitSchema, Transitions: []Transition{
		tr(TransitionCreate, nil, []WIPUnitID{one}), bind,
		tr(TransitionHandoff, []WIPUnitID{one}, []WIPUnitID{one}), tr(TransitionPark, []WIPUnitID{one}, []WIPUnitID{one}), tr(TransitionRecover, []WIPUnitID{one}, []WIPUnitID{one}),
		tr(TransitionCreate, nil, []WIPUnitID{two}), tr(TransitionCreate, nil, []WIPUnitID{three}), tr(TransitionSplit, []WIPUnitID{one}, []WIPUnitID{two, three}),
		tr(TransitionCreate, nil, []WIPUnitID{four}), tr(TransitionMerge, []WIPUnitID{two, three}, []WIPUnitID{four}),
		tr(TransitionCreate, nil, []WIPUnitID{five}), tr(TransitionSupersede, []WIPUnitID{four}, []WIPUnitID{five}), tr(TransitionLand, []WIPUnitID{five}, []WIPUnitID{five}),
	}}
	if err := ValidateHistory(history); err != nil {
		t.Fatal(err)
	}

	abandoned := History{Schema: WIPUnitSchema, Transitions: []Transition{tr(TransitionCreate, nil, []WIPUnitID{one}), tr(TransitionAbandon, []WIPUnitID{one}, []WIPUnitID{one})}}
	if err := ValidateHistory(abandoned); err != nil {
		t.Fatal(err)
	}
}

func TestIllegalTransitions(t *testing.T) {
	one, two, three := id(1), id(2), id(3)
	cases := []struct {
		name, want  string
		transitions []Transition
		schema      string
	}{
		{"incompatible version", "incompatible", nil, "fak-wip-unit/2"},
		{"unknown ID", "unknown", []Transition{tr(TransitionLand, []WIPUnitID{one}, []WIPUnitID{one})}, WIPUnitSchema},
		{"duplicate creation", "duplicate creation", []Transition{tr(TransitionCreate, nil, []WIPUnitID{one}), tr(TransitionCreate, nil, []WIPUnitID{one})}, WIPUnitSchema},
		{"terminal transition", "terminal", []Transition{tr(TransitionCreate, nil, []WIPUnitID{one}), tr(TransitionLand, []WIPUnitID{one}, []WIPUnitID{one}), tr(TransitionRecover, []WIPUnitID{one}, []WIPUnitID{one})}, WIPUnitSchema},
		{"self supersession", "self-supersession", []Transition{tr(TransitionCreate, nil, []WIPUnitID{one}), tr(TransitionSupersede, []WIPUnitID{one}, []WIPUnitID{one})}, WIPUnitSchema},
		{"malformed split", "split requires", []Transition{tr(TransitionCreate, nil, []WIPUnitID{one}), tr(TransitionSplit, []WIPUnitID{one}, []WIPUnitID{two})}, WIPUnitSchema},
		{"malformed merge", "merge requires", []Transition{tr(TransitionMerge, []WIPUnitID{one}, []WIPUnitID{two})}, WIPUnitSchema},
		{"cycle", "cycle", []Transition{tr(TransitionCreate, nil, []WIPUnitID{one}), tr(TransitionCreate, nil, []WIPUnitID{two}), tr(TransitionCreate, nil, []WIPUnitID{three}), tr(TransitionSplit, []WIPUnitID{one}, []WIPUnitID{two, three}), tr(TransitionMerge, []WIPUnitID{two, three}, []WIPUnitID{one})}, WIPUnitSchema},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateHistory(History{Schema: tc.schema, Transitions: tc.transitions})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want error containing %q", err, tc.want)
			}
		})
	}
}
