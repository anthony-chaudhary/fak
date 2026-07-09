package repoguard

import (
	"strings"
	"testing"
)

// The workspace used across these cases; its drive-stripped form is /work/fak,
// the exact string that failed in the 2026-07-09 trajectory audit (d5b3ed57).
const cdWS = "C:/work/fak"

func TestClassifyWorkspaceCdFlagsDriveStrippedRoot(t *testing.T) {
	cases := []string{
		"cd /work/fak",                         // the bare audit form
		"cd /work/fak && go test ./...",        // chained under && (the wasted-turn shape)
		"cd /work/fak/",                        // trailing slash
		"cd \"/work/fak\"",                      // quoted
		"cd '/work/fak'",                       // single-quoted
		"make ci; cd /work/fak && ls",          // second segment
	}
	for _, cmd := range cases {
		vs := ClassifyWorkspaceCd(cmd, cdWS)
		if len(vs) != 1 {
			t.Errorf("ClassifyWorkspaceCd(%q) = %d violations, want 1: %+v", cmd, len(vs), vs)
			continue
		}
		if vs[0].Reason != ReasonWorkspacePathUnmapped {
			t.Errorf("ClassifyWorkspaceCd(%q) reason = %q, want %q", cmd, vs[0].Reason, ReasonWorkspacePathUnmapped)
		}
		if !strings.Contains(vs[0].Fix, "C:/work/fak") {
			t.Errorf("ClassifyWorkspaceCd(%q) fix should name the host path: %q", cmd, vs[0].Fix)
		}
	}
}

func TestClassifyWorkspaceCdPassesCorrectAndUnrelatedForms(t *testing.T) {
	cases := []struct{ cmd, ws string }{
		{"cd C:/work/fak", cdWS},        // drive-ful host path — correct
		{"cd /c/work/fak", cdWS},        // MSYS form normalize() folds to the workspace — correct
		{"cd c:/work/fak && ls", cdWS},  // lower-drive is upcased to the workspace
		{"cd internal/repoguard", cdWS}, // in-tree relative cd
		{"cd ..", cdWS},                 // relative parent
		{"cd /work/other", cdWS},        // a different absolute path, not the workspace
		{"cd /work/fak-private", cdWS},  // sibling-shaped, not the root
		{"go test ./...", cdWS},         // no cd at all
		{"cd /work/fak extra", cdWS},    // two operands — not the bare mistake
		// A drive-less real-POSIX workspace: a leading-slash cd is legitimate and must never fire.
		{"cd /home/u/fak", "/home/u/fak"},
		{"cd /work/fak", "/home/u/fak"},
	}
	for _, c := range cases {
		if vs := ClassifyWorkspaceCd(c.cmd, c.ws); len(vs) != 0 {
			t.Errorf("ClassifyWorkspaceCd(%q, ws=%q) = %+v, want none", c.cmd, c.ws, vs)
		}
	}
}

func TestEvaluateWiresWorkspaceCdForBash(t *testing.T) {
	vs := Evaluate("Bash", map[string]any{"command": "cd /work/fak && go build ./..."}, cdWS, nil)
	if len(vs) != 1 || vs[0].Reason != ReasonWorkspacePathUnmapped {
		t.Fatalf("Evaluate(Bash, cd /work/fak) = %+v, want one WORKSPACE_PATH_UNMAPPED", vs)
	}
	// PowerShell parses differently; the POSIX cd rung must not fire on it.
	if vs := Evaluate("PowerShell", map[string]any{"command": "cd /work/fak"}, cdWS, nil); len(vs) != 0 {
		t.Errorf("Evaluate(PowerShell, cd /work/fak) = %+v, want none (POSIX rung is Bash-only)", vs)
	}
}

func TestWorkspaceCdDefaultSeverityIsWarn(t *testing.T) {
	if got := DefaultSeverity(ReasonWorkspacePathUnmapped); got != SeverityWarn {
		t.Errorf("DefaultSeverity(%s) = %v, want warn", ReasonWorkspacePathUnmapped, got)
	}
}

func TestRenderWorkspaceCdReasonNamesReasonAndFix(t *testing.T) {
	vs := ClassifyWorkspaceCd("cd /work/fak", cdWS)
	got := RenderReason(vs)
	for _, want := range []string{ReasonWorkspacePathUnmapped, "drive letter", "cd 'C:/work/fak'"} {
		if !strings.Contains(got, want) {
			t.Errorf("RenderReason() = %q, missing %q", got, want)
		}
	}
}
