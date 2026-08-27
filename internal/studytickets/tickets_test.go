package studytickets

import (
	"errors"
	"testing"
)

func ticket(id string, n int, deps ...string) Ticket {
	return Ticket{ClusterID: id, Disposition: Selected, Issue: n, Outcome: "bounded outcome", Source: "vllm#rev", For: "operators", Problem: "gap", Today: "manual", BetterBecause: "witnessed", Witness: "test", Centrality: "Core", P1P4: "pass", Horizon: "now", CloseCondition: "green receipt", NativeConstraint: "fak-native", Dependencies: deps}
}
func audit() Audit {
	return Audit{Schema: "fak.study-ticket-audit/1", Cutoff: "2026-08-26", SourceRevision: "vllm@abc", Checksum: "sha256:x", SourceCount: 4, CreatedCount: 1, ReusedCount: 1, Tickets: []Ticket{ticket("base", 100), ticket("next", 101, "base"), {ClusterID: "later", Disposition: Deferred}, {ClusterID: "old", Disposition: Landed}}}
}
func TestClosureAndDependencyQueue(t *testing.T) {
	a := audit()
	q, err := Queue(a)
	if err != nil {
		t.Fatal(err)
	}
	if len(q) != 2 || q[0].ClusterID != "base" || q[1].ClusterID != "next" {
		t.Fatalf("queue=%+v", q)
	}
}
func TestRejectsDuplicateMissingAndUnclassified(t *testing.T) {
	cases := []func(*Audit){func(a *Audit) { a.Tickets[1].Issue = 100 }, func(a *Audit) { a.SourceCount++ }, func(a *Audit) { a.Tickets[0].Witness = "" }, func(a *Audit) { a.Tickets[0].Dependencies = []string{"missing"} }}
	for _, mut := range cases {
		a := audit()
		mut(&a)
		if !errors.Is(Validate(a), ErrInvalid) {
			t.Fatal("invalid audit admitted")
		}
	}
}
