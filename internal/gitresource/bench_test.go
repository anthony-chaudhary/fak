package gitresource

import "testing"

func TestGitResourceBenchmarkSanity(t *testing.T) {
	h := validWorkspace()
	if err := h.Validate(); err != nil {
		t.Fatalf("valid workspace handle rejected: %v", err)
	}
}

func BenchmarkGitResource(b *testing.B) {
	h := validWorkspace()
	res := worktreeResource(WorktreeFiles, "worktree:bench")
	l := Lease{
		ID:       "lease:bench",
		Resource: res,
		Owner:    "owner:bench",
		Mode:     ExclusiveLease,
		State:    LeaseActive,
		Epoch:    1,
	}
	candidate := terminalCandidate()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		if err := h.Validate(); err != nil {
			b.Fatalf("workspace validate: %v", err)
		}
		if err := res.Validate(); err != nil {
			b.Fatalf("resource validate: %v", err)
		}
		if err := l.Validate(); err != nil {
			b.Fatalf("lease validate: %v", err)
		}
		if err := l.ValidateMutation(); err != nil {
			b.Fatalf("mutation validate: %v", err)
		}
		decision := AdmitCleanup(candidate)
		if !decision.Admitted {
			b.Fatalf("admit cleanup failed: %v", decision.Reason)
		}
	}
}
