package depthadmit

import (
	"testing"
)

func BenchmarkDepthAdmit(b *testing.B) {
	in := Input{
		Plan: []Phase{
			{ID: "p1", Title: "Bootstrap kernel foundation"},
			{ID: "p2", Title: "Wire admission boundary"},
			{ID: "p3", Title: "Enforce fail-closed guard"},
			{ID: "p4", Title: "Verify depth frontier"},
		},
		Witnessed: []string{"p1", "p2"},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Admit(in, ClosureMet)
	}
}

func BenchmarkDepthFold(b *testing.B) {
	in := Input{
		Plan: []Phase{
			{ID: "p1", Title: "Bootstrap kernel foundation"},
			{ID: "p2", Title: "Wire admission boundary"},
			{ID: "p3", Title: "Enforce fail-closed guard"},
			{ID: "p4", Title: "Verify depth frontier"},
		},
		Witnessed: []string{"p1", "p2", "foreign-tag"},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Fold(in)
	}
}

func BenchmarkDepthPersist(b *testing.B) {
	earlier := Fold(Input{
		Plan:      []Phase{{ID: "p1"}, {ID: "p2"}, {ID: "p3"}},
		Witnessed: []string{"p1"},
	})
	later := Fold(Input{
		Plan:      []Phase{{ID: "p1"}, {ID: "p2"}, {ID: "p3"}},
		Witnessed: []string{"p1", "p2"},
	})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Persist(earlier, later)
	}
}

func TestBenchmarkSanity(t *testing.T) {
	in := Input{
		Plan:      []Phase{{ID: "p1"}, {ID: "p2"}},
		Witnessed: []string{"p1"},
	}
	d := Admit(in, ClosureMet)
	if d.Admitted {
		t.Fatal("expected uncarried plan to be refused")
	}
	if d.Reason != RefusalReason {
		t.Fatalf("expected refusal reason %s, got %s", RefusalReason, d.Reason)
	}
}
