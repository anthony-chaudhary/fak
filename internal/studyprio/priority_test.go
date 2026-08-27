package studyprio

import (
	"errors"
	"reflect"
	"testing"
)

func c(id string, score int, deps ...string) Candidate {
	return Candidate{ID: id, Category: "native", Horizon: "now", Scores: Scores{Centrality: score, NativeImpact: 5, EndToEndValue: 5, Evidence: 3, ImplementationCost: 2}, HardGatePass: true, Dependencies: deps, Frame: Frame{For: "operators", Problem: "memory traffic", Today: "baseline", BetterBecause: "less traffic", Witness: "accepted-token receipt", Centrality: "Core", P1P4: "pass"}}
}
func TestDeterministicDependencyQueueAndGates(t *testing.T) {
	v := []Candidate{c("base", 10), c("dependent", 20, "base"), c("small", 1)}
	v[2].HardGatePass = false
	v[2].GateReason = "no native witness"
	a, err := Rank(v)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := Rank(v)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("nondeterministic")
	}
	if a[0].Candidate.ID != "base" || a[1].Candidate.ID != "dependent" || a[2].Priority != "rejected" {
		t.Fatalf("queue=%+v", a)
	}
}
func TestRejectsCyclesMissingFramesAndDependencies(t *testing.T) {
	cases := [][]Candidate{{c("a", 1, "b"), c("b", 1, "a")}, {c("a", 1, "missing")}, {func() Candidate { x := c("a", 1); x.Frame.Witness = ""; return x }()}}
	for _, v := range cases {
		if _, err := Rank(v); !errors.Is(err, ErrInvalid) {
			t.Fatal("invalid queue admitted")
		}
	}
}
