package studytickets

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"
)

func ticket(id string, n int, deps ...string) Ticket {
	return Ticket{ClusterID: id, CandidateID: id, Disposition: Selected, Issue: n, Outcome: "bounded", Source: "vllm@rev", For: "operators", Problem: "gap", Today: "manual", BetterBecause: "witnessed", Witness: "test", Centrality: "Core", P1P4: "pass", Horizon: "now", CloseCondition: "green", NativeConstraint: "fak-native", Dependencies: deps}
}
func audit() Audit {
	return Audit{Schema: "fak.study-ticket-audit/1", Cutoff: "2026-08-26", SourceRevision: "vllm@abc", Checksum: "sha256:x", PriorityChecksum: "sha256:y", SourceCount: 4, CreatedCount: 1, ReusedCount: 1, RefreshObligations: []string{"refresh"}, Tickets: []Ticket{ticket("base", 100), func() Ticket { x := ticket("next", 101, "base"); x.Existing = true; return x }(), {ClusterID: "later", Disposition: Deferred}, {ClusterID: "old", Disposition: Landed}}}
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
func TestRejectsInvalidAudit(t *testing.T) {
	cases := []func(*Audit){func(a *Audit) { a.Tickets[1].Issue = 100 }, func(a *Audit) { a.SourceCount++ }, func(a *Audit) { a.Tickets[0].Witness = "" }, func(a *Audit) { a.Tickets[0].Dependencies = []string{"missing"} }, func(a *Audit) { a.Tickets[2].Disposition = "unknown" }, func(a *Audit) { a.CreatedCount++ }, func(a *Audit) { a.RefreshObligations = nil }}
	for i, mut := range cases {
		a := audit()
		mut(&a)
		if !errors.Is(Validate(a), ErrInvalid) {
			t.Fatalf("case %d admitted", i)
		}
	}
}
func TestRealClosureLedger(t *testing.T) {
	a, err := LoadClosureLedger()
	if err != nil {
		t.Fatal(err)
	}
	s, err := Summary(a)
	if err != nil {
		t.Fatal(err)
	}
	if s.SourceClusters != 5 || s.ClassifiedClusters != 5 || s.QueueTickets != 5 || s.Created != 2 || s.Reused != 0 {
		t.Fatalf("summary=%+v", s)
	}
	if a.Checksum != "sha256:2e7b2a6c64a3b2a48a6a49db0bed2086e82327d4b41c150b3fa297f424b6413d" || a.PriorityChecksum != "sha256:df1f2114211a78b15f88decad3bb02de1e3ff790e150bd65c8ec55b73d68a417" {
		t.Fatal("source provenance drift")
	}
	want := map[string]int{"architecture_runtime:body:vllm-ir": 9377, "architecture_runtime:label:vllm-ir": 9377, "architecture_runtime:title:vllm-ir": 9377, "kernels_compilation:label:vllm-ir": 9377, "memory_residency:body:allocator-fragmentation": 9378}
	got := map[string]int{}
	for _, x := range a.Tickets {
		got[x.ClusterID] = x.Issue
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("mapping=%v", got)
	}
	q, err := Queue(a)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 4; i++ {
		if q[i].Issue != 9377 {
			t.Fatalf("queue[%d]=%+v", i, q[i])
		}
	}
	if q[4].Issue != 9378 {
		t.Fatalf("queue tail=%+v", q[4])
	}
	raw, err := json.Marshal(a)
	if err != nil || len(raw) == 0 {
		t.Fatalf("marshal: %v", err)
	}
}
