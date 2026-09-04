package dosadapter

import (
	"errors"
	"testing"
)

func TestPathMatchesPattern(t *testing.T) {
	tests := []struct {
		pattern string
		path    string
		want    bool
	}{
		{
			pattern: "internal/gateway/**",
			path:    "internal/gateway/mcp.go",
			want:    true,
		},
		{
			pattern: "internal/gateway/**",
			path:    "internal/gateway/sub/deep/file.go",
			want:    true,
		},
		{
			pattern: "internal/gateway/**",
			path:    "internal/gateway",
			want:    true,
		},
		{
			pattern: "internal/gateway/**",
			path:    "internal/dosadapter/dosadapter.go",
			want:    false,
		},
		{
			pattern: "internal/dosadapter/dosadapter.go",
			path:    "internal/dosadapter/dosadapter.go",
			want:    true,
		},
		{
			pattern: "internal/dosadapter/dosadapter.go",
			path:    "internal/dosadapter/other.go",
			want:    false,
		},
		{
			pattern: "cmd/fak/*",
			path:    "cmd/fak/main.go",
			want:    true,
		},
		{
			pattern: "cmd/fak/*",
			path:    "cmd/fak/nested/deep.go",
			want:    false,
		},
		{
			pattern: "internal/gateway/**",
			path:    "internal\\gateway\\mcp.go", // Windows backslash normalization
			want:    true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.pattern+"_vs_"+tc.path, func(t *testing.T) {
			got := PathMatchesPattern(tc.pattern, tc.path)
			if got != tc.want {
				t.Errorf("PathMatchesPattern(%q, %q) = %v, want %v", tc.pattern, tc.path, got, tc.want)
			}
		})
	}
}

func TestPatternsOverlap(t *testing.T) {
	tests := []struct {
		patA string
		patB string
		want bool
	}{
		{
			patA: "internal/gateway/**",
			patB: "internal/dosadapter/**",
			want: false,
		},
		{
			patA: "internal/gateway/**",
			patB: "internal/gateway/**",
			want: true,
		},
		{
			patA: "internal/**",
			patB: "internal/gateway/**",
			want: true,
		},
		{
			patA: "internal/gateway/**",
			patB: "internal/**",
			want: true,
		},
		{
			patA: "internal/gateway/**",
			patB: "internal/gateway/mcp.go",
			want: true,
		},
		{
			patA: "internal/gateway/mcp.go",
			patB: "internal/gateway/**",
			want: true,
		},
		{
			patA: "cmd/fak/main.go",
			patB: "cmd/fak/other.go",
			want: false,
		},
		{
			patA: "cmd/fak/*",
			patB: "cmd/fak/main.go",
			want: true,
		},
		{
			patA: "cmd/fak/main.go",
			patB: "cmd/fak/*",
			want: true,
		},
		{
			patA: "docs/**",
			patB: "internal/**",
			want: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.patA+"_and_"+tc.patB, func(t *testing.T) {
			got := PatternsOverlap(tc.patA, tc.patB)
			if got != tc.want {
				t.Errorf("PatternsOverlap(%q, %q) = %v, want %v", tc.patA, tc.patB, got, tc.want)
			}
		})
	}
}

func TestTreesOverlap(t *testing.T) {
	treeA := []string{"internal/gateway/**", "cmd/fak/**"}
	treeB := []string{"internal/dosadapter/**", "cmd/fak/main.go"}
	treeC := []string{"docs/**", "examples/**"}

	if !TreesOverlap(treeA, treeB) {
		t.Errorf("TreesOverlap(treeA, treeB) = false, want true (intersect on cmd/fak)")
	}
	if TreesOverlap(treeA, treeC) {
		t.Errorf("TreesOverlap(treeA, treeC) = true, want false")
	}
	if TreesOverlap([]string{}, treeA) {
		t.Errorf("TreesOverlap with empty tree = true, want false")
	}
}

func TestLockModeInteractions(t *testing.T) {
	activeShared := LeaseRequest{
		ID:       "held-shared",
		Lane:     "docs",
		LockMode: LockModeShared,
		Tree:     []string{"docs/**"},
	}

	activeExclusive := LeaseRequest{
		ID:       "held-exclusive",
		Lane:     "gateway",
		LockMode: LockModeExclusive,
		Tree:     []string{"internal/gateway/**"},
	}

	t.Run("shared request over shared active lease admits", func(t *testing.T) {
		req := LeaseRequest{
			ID:       "new-shared",
			Lane:     "docs-review",
			LockMode: LockModeShared,
			Tree:     []string{"docs/**"},
		}
		if err := CheckDisjoint(req, []LeaseRequest{activeShared}); err != nil {
			t.Fatalf("CheckDisjoint() unexpected error: %v (shared/shared should overlap)", err)
		}
	})

	t.Run("exclusive request over shared active lease refuses", func(t *testing.T) {
		req := LeaseRequest{
			ID:       "new-exclusive-on-shared",
			Lane:     "docs-reorg",
			LockMode: LockModeExclusive,
			Tree:     []string{"docs/**"},
		}
		err := CheckDisjoint(req, []LeaseRequest{activeShared})
		if err == nil {
			t.Fatalf("CheckDisjoint() expected error, got nil")
		}
		if !errors.Is(err, ErrDisjointnessViolation) {
			t.Errorf("CheckDisjoint() error = %v, want ErrDisjointnessViolation", err)
		}
	})

	t.Run("shared request over exclusive active lease refuses", func(t *testing.T) {
		req := LeaseRequest{
			ID:       "new-shared-on-exclusive",
			Lane:     "gateway-read",
			LockMode: LockModeShared,
			Tree:     []string{"internal/gateway/mcp.go"},
		}
		err := CheckDisjoint(req, []LeaseRequest{activeExclusive})
		if err == nil {
			t.Fatalf("CheckDisjoint() expected error, got nil")
		}
		if !errors.Is(err, ErrDisjointnessViolation) {
			t.Errorf("CheckDisjoint() error = %v, want ErrDisjointnessViolation", err)
		}
	})

	t.Run("exclusive request over exclusive active lease refuses on overlap", func(t *testing.T) {
		req := LeaseRequest{
			ID:       "new-exclusive-collision",
			Lane:     "gateway-edit",
			LockMode: LockModeExclusive,
			Tree:     []string{"internal/gateway/mcp.go"},
		}
		err := CheckDisjoint(req, []LeaseRequest{activeExclusive})
		if err == nil {
			t.Fatalf("CheckDisjoint() expected error, got nil")
		}
		if !errors.Is(err, ErrDisjointnessViolation) {
			t.Errorf("CheckDisjoint() error = %v, want ErrDisjointnessViolation", err)
		}
	})

	t.Run("exclusive request over exclusive active lease admits on disjoint tree", func(t *testing.T) {
		req := LeaseRequest{
			ID:       "new-exclusive-disjoint",
			Lane:     "dosadapter",
			LockMode: LockModeExclusive,
			Tree:     []string{"internal/dosadapter/**"},
		}
		if err := CheckDisjoint(req, []LeaseRequest{activeExclusive}); err != nil {
			t.Fatalf("CheckDisjoint() unexpected error: %v", err)
		}
	})

	t.Run("self lease ID does not conflict during re-arbitration", func(t *testing.T) {
		req := LeaseRequest{
			ID:       "held-exclusive", // same ID
			Lane:     "gateway",
			LockMode: LockModeExclusive,
			Tree:     []string{"internal/gateway/**"},
		}
		if err := CheckDisjoint(req, []LeaseRequest{activeExclusive}); err != nil {
			t.Fatalf("CheckDisjoint() unexpected self-conflict error: %v", err)
		}
	})
}

func TestArbitrationEdgeCases(t *testing.T) {
	t.Run("empty active leases slice always passes disjointness", func(t *testing.T) {
		req := LeaseRequest{
			ID:       "first-lease",
			Lane:     "gateway",
			LockMode: LockModeExclusive,
			Tree:     []string{"internal/gateway/**"},
		}
		if err := CheckDisjoint(req, nil); err != nil {
			t.Errorf("CheckDisjoint() with nil active leases error = %v, want nil", err)
		}
		if err := CheckDisjoint(req, []LeaseRequest{}); err != nil {
			t.Errorf("CheckDisjoint() with empty active leases error = %v, want nil", err)
		}
	})

	t.Run("multiple trees in request with partial overlap", func(t *testing.T) {
		active := LeaseRequest{
			ID:       "held-gateway",
			Lane:     "gateway",
			LockMode: LockModeExclusive,
			Tree:     []string{"internal/gateway/**"},
		}
		req := LeaseRequest{
			ID:       "multi-tree-req",
			Lane:     "multi",
			LockMode: LockModeExclusive,
			Tree:     []string{"internal/dosadapter/**", "internal/gateway/debug.go"},
		}
		err := CheckDisjoint(req, []LeaseRequest{active})
		if !errors.Is(err, ErrDisjointnessViolation) {
			t.Errorf("CheckDisjoint() with partial overlap error = %v, want ErrDisjointnessViolation", err)
		}
	})

	t.Run("lock mode case insensitivity", func(t *testing.T) {
		activeShared := LeaseRequest{
			ID:       "active-sh",
			Lane:     "docs",
			LockMode: "SHARED",
			Tree:     []string{"docs/**"},
		}
		reqShared := LeaseRequest{
			ID:       "req-sh",
			Lane:     "docs-two",
			LockMode: "shared",
			Tree:     []string{"docs/**"},
		}
		if err := CheckDisjoint(reqShared, []LeaseRequest{activeShared}); err != nil {
			t.Errorf("CheckDisjoint() case insensitive shared/shared failed: %v", err)
		}
	})
}
