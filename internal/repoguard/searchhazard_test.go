package repoguard

import (
	"strings"
	"testing"
)

// The selftest fixture workspace root (cmd/repoguard/main.go runSelftest).
const shWS = "C:/Users/u/work/fak"

func TestClassifySearchHazardFlagsRecursiveAtBroadRoot(t *testing.T) {
	cases := []struct {
		cmd    string
		wantOp string
	}{
		// The witnessed grep-spam form: recursive grep from the repo root.
		{"grep -rn foo .", "grep -r"},
		{"grep -r foo .", "grep -r"},
		{"grep -R foo .", "grep -r"},
		{"grep --recursive foo .", "grep -r"},
		{"grep -rln foo .", "grep -r"},
		{"grep -rl foo .", "grep -r"},
		// An absolute workspace root is the same broad root.
		{"grep -r foo C:/Users/u/work/fak", "grep -r"},
		{"grep -r foo /c/Users/u/work/fak", "grep -r"}, // MSYS drive form
		// An ancestor of the workspace (the walk includes .worktrees).
		{"grep -r foo C:/Users/u/work", "grep -r"},
		{"grep -r foo ..", "grep -r"}, // .. from the workspace root
		// A broad drive / POSIX / home root.
		{"grep -r foo C:/", "grep -r"},
		{"grep -r foo /", "grep -r"},
		{"grep -r foo C:/Users/u", "grep -r"},
		// The .worktrees farm directly.
		{"grep -r foo .worktrees", "grep -r"},
		{"grep -r foo .worktrees/", "grep -r"},
		// find at a broad root without -maxdepth.
		{"find . -name '*.go'", "find"},
		{"find . -type f", "find"},
		{"find C:/Users/u/work/fak -name '*.go'", "find"},
		{"find .worktrees -name '*.go'", "find"},
	}
	for _, tc := range cases {
		vs := ClassifySearchHazard(tc.cmd, shWS)
		if len(vs) != 1 {
			t.Errorf("ClassifySearchHazard(%q) = %d violations, want 1: %+v", tc.cmd, len(vs), vs)
			continue
		}
		if vs[0].Reason != ReasonSearchHazard {
			t.Errorf("ClassifySearchHazard(%q) reason = %q, want %q", tc.cmd, vs[0].Reason, ReasonSearchHazard)
		}
		if vs[0].Op != tc.wantOp {
			t.Errorf("ClassifySearchHazard(%q) op = %q, want %q", tc.cmd, vs[0].Op, tc.wantOp)
		}
		if vs[0].Fix == "" {
			t.Errorf("ClassifySearchHazard(%q) fix empty, want a steering hint", tc.cmd)
		}
	}
}

func TestClassifySearchHazardAllowsScopedAndBenign(t *testing.T) {
	cases := []string{
		// Non-recursive grep is never the hazard.
		"grep -o ../foo internal/policy/x.go",
		"grep foo .",
		"grep -n foo .",
		// Recursive grep scoped to a subdirectory.
		"grep -r foo internal/",
		"grep -rn foo cmd/fak",
		"grep -r foo internal/repoguard",
		// Backgrounded: does not hold the turn.
		"grep -r foo . &",
		"grep -rn foo . &",
		// find scoped to a subdirectory.
		"find internal -name '*.go'",
		"find cmd/fak -name '*.go'",
		// find bounded by -maxdepth.
		"find . -maxdepth 2 -name '*.go'",
		"find . -maxdepth 1 -type f",
		// ripgrep honors .gitignore — not this rung's concern.
		"rg -r foo .",
		"rg foo .",
		// A non-search verb.
		"ls -R .",
	}
	for _, c := range cases {
		if vs := ClassifySearchHazard(c, shWS); len(vs) != 0 {
			t.Errorf("ClassifySearchHazard(%q) = %d violations, want 0: %+v", c, len(vs), vs)
		}
	}
}

// TestClassifySearchHazardUnresolvableStaysOut confirms a glob / $var / ~ search
// root is left alone (toAbs rejects it) rather than guessed.
func TestClassifySearchHazardUnresolvableStaysOut(t *testing.T) {
	for _, c := range []string{
		"grep -r foo $ROOT",
		"grep -r foo ~/work",
		"grep -r foo ./*",
	} {
		if vs := ClassifySearchHazard(c, shWS); len(vs) != 0 {
			t.Errorf("ClassifySearchHazard(%q) = %d violations, want 0 (unresolvable root): %+v", c, len(vs), vs)
		}
	}
}

// TestSearchHazardRenderReason confirms the advisory block names the reason and
// the per-reason silence dial, matching the other fix-bearing rungs.
func TestSearchHazardRenderReason(t *testing.T) {
	vs := ClassifySearchHazard("grep -rn foo .", shWS)
	if len(vs) != 1 {
		t.Fatalf("want 1 violation, got %d", len(vs))
	}
	got := renderReason(vs)
	for _, want := range []string{ReasonSearchHazard, "grep -r", "FAK_REPO_GUARD_SEVERITY=" + ReasonSearchHazard} {
		if !strings.Contains(got, want) {
			t.Errorf("renderReason = %q, want it to contain %q", got, want)
		}
	}
}
