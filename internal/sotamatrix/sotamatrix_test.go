package sotamatrix

import (
	"path"
	"strings"
	"testing"
)

// anyGlobMatches mirrors the PRIOR_ART gate's path.Match coverage semantics closely
// enough to assert that a row covers a specific kernel file: both operate on
// forward-slash paths and no glob in the matrix needs to cross a separator.
func anyGlobMatches(globs []string, p string) bool {
	for _, g := range globs {
		if ok, _ := path.Match(g, p); ok {
			return true
		}
	}
	return false
}

// TestCollectiveCommRowCoversNCCLProcessGroup pins the fix for a SOTA-matrix blind spot:
// internal/compute/cuda_nccl_pg.cu (the multi-PROCESS NCCL bootstrap) was covered by no
// prior-art row, so an agent optimizing the collective could re-derive a ring/tree
// all-reduce with no reference or oracle in hand. This asserts the collective-comm row
// exists and that its FileGlobs actually cover the previously-orphaned kernel file. RED
// before the row was added (BySlug fails), GREEN after.
func TestCollectiveCommRowCoversNCCLProcessGroup(t *testing.T) {
	op, ok := BySlug("collective-comm")
	if !ok {
		t.Fatal("collective-comm row missing from the SOTA matrix")
	}
	const orphan = "internal/compute/cuda_nccl_pg.cu"
	if !anyGlobMatches(op.FileGlobs, orphan) {
		t.Fatalf("collective-comm FileGlobs %v do not cover %s", op.FileGlobs, orphan)
	}
	// The device file also binds -lnccl and calls NCCL directly; the honest route is bind,
	// not a from-scratch all-reduce.
	if op.Route != RouteBind {
		t.Errorf("collective-comm route = %q, want %q", op.Route, RouteBind)
	}
}

// TestEveryRowResolvesAndIsComplete mirrors the coverage scorecard's per-row HARD KPIs
// in-binary: every row resolves by its own slug and carries a primary http(s) link, an
// oracle, and a fak-path. A row that fails this is a matrix that silently stopped being
// load-bearing — the exact rot the datum exists to prevent.
func TestEveryRowResolvesAndIsComplete(t *testing.T) {
	seen := map[string]bool{}
	for _, op := range Operations() {
		if seen[op.Slug] {
			t.Errorf("duplicate slug %q", op.Slug)
		}
		seen[op.Slug] = true
		if got, ok := BySlug(op.Slug); !ok || got.Slug != op.Slug {
			t.Errorf("row %q does not resolve via BySlug", op.Slug)
		}
		if !strings.HasPrefix(op.PrimaryLink, "http") {
			t.Errorf("row %q PrimaryLink = %q, want an http(s) SOTA link", op.Slug, op.PrimaryLink)
		}
		if strings.TrimSpace(op.Oracle) == "" {
			t.Errorf("row %q has no oracle (a kernel with no oracle is not done)", op.Slug)
		}
		if strings.TrimSpace(op.FakPath) == "" {
			t.Errorf("row %q has no FakPath", op.Slug)
		}
	}
}
