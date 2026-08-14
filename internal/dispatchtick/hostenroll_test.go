package dispatchtick

import "testing"

// TestPlanHostEnrollmentCopiesLeaseTreeAndDerivesStableID pins the pure descriptor:
// the per-issue agent id, the verbatim (copied, not aliased) lease-fence tree that
// makes M11 disjointness hold by construction, and the empty-lease fallback.
func TestPlanHostEnrollmentCopiesLeaseTreeAndDerivesStableID(t *testing.T) {
	tree := []string{"docs/**"}
	plan := PlanHostEnrollment("docs", 12, "resolve-docs", tree)
	if plan.Lane != "docs" || plan.Issue != 12 || plan.LeaseID != "resolve-docs" {
		t.Fatalf("plan identity = %+v", plan)
	}
	if plan.AgentID != "resolve-docs-12" {
		t.Fatalf("agent id = %q, want resolve-docs-12", plan.AgentID)
	}
	// A NARROWED lease id already ends in "-<issue>"; the agent id must stay idempotent
	// and NOT double the suffix (resolve-docs-12-12) across the coarse and narrowed
	// (wave/tick per-issue) lease grammars.
	if got := PlanHostEnrollment("docs", 12, "resolve-docs-12", tree).AgentID; got != "resolve-docs-12" {
		t.Fatalf("narrowed agent id = %q, want resolve-docs-12 (no doubled suffix)", got)
	}
	if len(plan.Tree) != 1 || plan.Tree[0] != "docs/**" {
		t.Fatalf("plan tree = %v, want [docs/**]", plan.Tree)
	}
	// The fence tree must be a COPY: mutating the caller's slice cannot reach the plan.
	tree[0] = "MUTATED"
	if plan.Tree[0] != "docs/**" {
		t.Fatalf("plan tree aliased the caller's slice: %v", plan.Tree)
	}
	// Empty lease id falls back to a lane-derived, id-safe token (path runes folded).
	if got := PlanHostEnrollment("internal/foo", 7, "", nil).AgentID; got != "resolve-internal-foo-7" {
		t.Fatalf("fallback agent id = %q, want resolve-internal-foo-7", got)
	}
}

// TestIsMicroBackendAndNormalize pins that micro is a recognized, case-insensitive
// backend token (the #2030 config selector) alongside the CLI backends.
func TestIsMicroBackendAndNormalize(t *testing.T) {
	for _, s := range []string{"micro", "MICRO", " micro "} {
		if !IsMicroBackend(s) {
			t.Errorf("IsMicroBackend(%q) = false, want true", s)
		}
	}
	for _, s := range []string{"claude", "opencode", "codex", ""} {
		if IsMicroBackend(s) {
			t.Errorf("IsMicroBackend(%q) = true, want false", s)
		}
	}
	if got, err := NormalizeBackend("MICRO"); err != nil || got != MicroBackend {
		t.Fatalf("NormalizeBackend(MICRO) = %q, %v; want micro, nil", got, err)
	}
}

func TestProductForMicroBackendIsLocal(t *testing.T) {
	if got := ProductForBackend("micro"); got != MicroBackend {
		t.Fatalf("ProductForBackend(micro) = %q, want %q", got, MicroBackend)
	}
}
