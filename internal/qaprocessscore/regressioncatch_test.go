package qaprocessscore

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/brittleness"
)

// c is a terse fixture-commit builder (newest-first windows read top-to-bottom).
func c(sha, subject string, files ...string) brittleness.Commit {
	return brittleness.Commit{SHA: sha, Subject: subject, Files: files}
}

func TestRegressionCatch(t *testing.T) {
	tests := []struct {
		name        string
		commits     []brittleness.Commit
		wantScore   float64
		wantDefects int
		wantRef     string // a SHA the (single) defect line must name; "" = expect no defects
	}{
		{
			name: "revert with no accompanying test is a gap",
			commits: []brittleness.Commit{
				c("aaa1", `Revert "feat(foo): risky change"`, "internal/foo/foo.go"),
			},
			wantScore:   0,
			wantDefects: 1,
			wantRef:     "aaa1",
		},
		{
			name: "revert bundling a same-package test is caught",
			commits: []brittleness.Commit{
				c("bbb1", `Revert "feat(foo): risky change"`, "internal/foo/foo.go", "internal/foo/foo_test.go"),
			},
			wantScore:   100,
			wantDefects: 0,
		},
		{
			name: "test in a NEWER follow-up commit catches the revert",
			commits: []brittleness.Commit{
				c("ccc0", "test(bar): add regression test for the reverted bug", "internal/bar/bar_test.go"),
				c("ccc1", "revert(bar): undo bar change", "internal/bar/bar.go"),
			},
			wantScore:   100,
			wantDefects: 0,
		},
		{
			name: "test in an OLDER commit does NOT count (predates the revert)",
			commits: []brittleness.Commit{
				c("ddd0", `Revert "feat(baz): risky"`, "internal/baz/baz.go"),
				c("ddd1", "test(baz): unrelated earlier test", "internal/baz/baz_test.go"),
			},
			wantScore:   0,
			wantDefects: 1,
			wantRef:     "ddd0",
		},
		{
			name: "non-revert commits are never examined",
			commits: []brittleness.Commit{
				c("eee1", "feat(qux): add feature with no test", "internal/qux/qux.go"),
			},
			wantScore:   100,
			wantDefects: 0,
		},
		{
			name: "a revert touching no Go code (docs only) is not examined",
			commits: []brittleness.Commit{
				c("fff1", `Revert "docs: tweak readme"`, "README.md", "docs/x.md"),
			},
			wantScore:   100,
			wantDefects: 0,
		},
		{
			name: "a revert touching only test files is not examined (no production behavior reverted)",
			commits: []brittleness.Commit{
				c("999a", `Revert "test(z): flaky test"`, "internal/z/z_test.go"),
			},
			wantScore:   100,
			wantDefects: 0,
		},
		{
			name: "mixed: one caught, one gap -> 50% and one defect",
			commits: []brittleness.Commit{
				c("h001", `Revert "feat(a): A"`, "internal/a/a.go", "internal/a/a_test.go"), // caught (bundled test)
				c("h002", `Revert "feat(b): B"`, "internal/b/b.go"),                         // gap
			},
			wantScore:   50,
			wantDefects: 1,
			wantRef:     "h002",
		},
		{
			// The heuristic is revert-level, not per-package: a revert that reverted code in
			// TWO packages but bundled a _test.go for only ONE of them still counts as
			// "caught" -- the revert got an accompanying regression test, which is all the
			// coarse "a _test.go touching the same package" signal asks for. (#3841)
			name: "multi-package revert: a test in ONE reverted package catches it (revert-level heuristic)",
			commits: []brittleness.Commit{
				c("m001", `Revert "feat(multi): touch two packages"`,
					"internal/a/a.go", "internal/a/a_test.go", "internal/b/b.go"),
			},
			wantScore:   100,
			wantDefects: 0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			kpi := RegressionCatch(tc.commits)
			if kpi.Key != RegressionCatchKey {
				t.Errorf("Key = %q, want %q", kpi.Key, RegressionCatchKey)
			}
			if kpi.Group != "regression" {
				t.Errorf("Group = %q, want %q", kpi.Group, "regression")
			}
			if kpi.Score != tc.wantScore {
				t.Errorf("Score = %v, want %v (detail: %q)", kpi.Score, tc.wantScore, kpi.Detail)
			}
			if len(kpi.Defects) != tc.wantDefects {
				t.Fatalf("got %d defects, want %d: %v", len(kpi.Defects), tc.wantDefects, kpi.Defects)
			}
			if tc.wantRef != "" {
				if !strings.Contains(kpi.Defects[0], tc.wantRef) {
					t.Errorf("defect %q does not name revert SHA %q", kpi.Defects[0], tc.wantRef)
				}
				if !strings.Contains(kpi.Defects[0], regressionGapClass) {
					t.Errorf("defect %q missing work-list class token %q", kpi.Defects[0], regressionGapClass)
				}
			}
		})
	}
}

// A revert that reverted production code in MORE THAN ONE package with no accompanying
// _test.go in any of them is one gap whose work item must name EVERY reverted package --
// the worst-first list has to point at all the coverage holes the revert exposed, not just
// the first. This exercises the multi-package Packages rendering that the single-package
// table fixtures never reach. (#3841)
func TestRegressionCatchGapNamesEveryRevertedPackage(t *testing.T) {
	kpi := RegressionCatch([]brittleness.Commit{
		c("m101", `Revert "feat(multi): risky two-package change"`,
			"internal/a/a.go", "internal/b/b.go"),
	})
	if kpi.Score != 0 {
		t.Errorf("Score = %v, want 0 (detail: %q)", kpi.Score, kpi.Detail)
	}
	if len(kpi.Defects) != 1 {
		t.Fatalf("got %d defects, want 1: %v", len(kpi.Defects), kpi.Defects)
	}
	for _, pkg := range []string{"internal/a", "internal/b"} {
		if !strings.Contains(kpi.Defects[0], pkg) {
			t.Errorf("defect %q does not name reverted package %q", kpi.Defects[0], pkg)
		}
	}
}

// A gap must never surface as SOFT: regression_catch is a HARD KPI, so a missing
// regression test is debt that counts, not an advisory nudge.
func TestRegressionCatchIsHardNotSoft(t *testing.T) {
	kpi := RegressionCatch([]brittleness.Commit{
		c("g001", `Revert "feat(foo): risky"`, "internal/foo/foo.go"),
	})
	if len(kpi.Soft) != 0 {
		t.Errorf("regression_catch must be HARD, but emitted SOFT items: %v", kpi.Soft)
	}
	if len(kpi.Defects) != 1 {
		t.Errorf("expected the gap as a HARD defect, got %d defects", len(kpi.Defects))
	}
}
