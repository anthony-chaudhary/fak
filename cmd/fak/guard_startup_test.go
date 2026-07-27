package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// scratchRootsHas reports whether the resolved FAK_GUARD_SCRATCHPAD_ROOTS list
// contains want as a whole entry (not a substring of a longer root).
func scratchRootsHas(got, want string) bool {
	for _, r := range strings.Split(got, string(os.PathListSeparator)) {
		if strings.EqualFold(strings.TrimSpace(r), want) {
			return true
		}
	}
	return false
}

func TestGuardCapabilityFloorDefaultsScratchpadRoot(t *testing.T) {
	t.Setenv("FAK_GUARD_SCRATCHPAD_ROOTS", "")
	loadGuardCapabilityFloor("")
	want := filepath.Join(os.TempDir(), "claude")
	got := os.Getenv("FAK_GUARD_SCRATCHPAD_ROOTS")
	if !scratchRootsHas(got, want) {
		t.Fatalf("scratchpad roots=%q want to contain %q", got, want)
	}
	// The default must stay the narrow Claude scratch tree — never the OS temp
	// directory itself, which every root here sits strictly below.
	for _, r := range strings.Split(got, string(os.PathListSeparator)) {
		if strings.EqualFold(strings.TrimRight(strings.ReplaceAll(r, `\`, "/"), "/"),
			strings.TrimRight(strings.ReplaceAll(os.TempDir(), `\`, "/"), "/")) {
			t.Fatalf("scratchpad roots=%q must not declare the OS temp directory itself", got)
		}
	}
}

func TestGuardCapabilityFloorPreservesScratchpadOverride(t *testing.T) {
	want := filepath.Join(t.TempDir(), "session-scratch")
	t.Setenv("FAK_GUARD_SCRATCHPAD_ROOTS", want)
	loadGuardCapabilityFloor("")
	if got := os.Getenv("FAK_GUARD_SCRATCHPAD_ROOTS"); !scratchRootsHas(got, want) {
		t.Fatalf("scratchpad override=%q want to contain %q", got, want)
	}
}

// TestScratchpadRootCarriesBothHostSpellings pins the fix for the defect that made
// the recursive-delete scratch carve-out the largest remaining refusal class in the
// guard-audit corpus (49 of 103 POLICY_BLOCKs, all dated after the carve-out
// shipped). The gates prove containment by string comparison, so a root declared
// only as `C:/…` could never match the `/c/…` spelling Git Bash — the shell behind
// the Bash tool — uses for that identical directory. A live probe of one throwaway
// directory inside the session scratchpad reproduced it: `/c/…` was hard-denied,
// byte-equivalent `C:/…` fell through to the preview-confirm gate.
func TestScratchpadRootCarriesBothHostSpellings(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("drive-letter/MSYS spelling duality is Windows-only")
	}
	got := guardScratchpadRootsValue(`C:\agent-scratch\claude`)
	for _, want := range []string{`C:\agent-scratch\claude`, "/c/agent-scratch/claude"} {
		if !scratchRootsHas(got, want) {
			t.Errorf("roots=%q want to contain %q", got, want)
		}
	}
	// Aliasing is symmetric: a root declared in the Git Bash spelling must also
	// cover a delete a PowerShell-backed surface spells with the drive letter.
	got = guardScratchpadRootsValue("/c/agent-scratch/claude")
	if !scratchRootsHas(got, "C:/agent-scratch/claude") {
		t.Errorf("roots=%q want to contain the drive-letter alias", got)
	}
}

// TestScratchpadRootAliasNeverAddsADirectory is the safety half: an alias is a
// second NAME for an already-declared root, never an extra directory. Every alias
// must therefore rewrite only the root prefix and keep the trailing path intact —
// if one ever mapped to a shorter or different tree it would silently widen the
// carve-out past what the operator declared.
func TestScratchpadRootAliasNeverAddsADirectory(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{`C:\agent-scratch\claude`, "/c/agent-scratch/claude"},
		{"C:/agent-scratch/claude/", "/c/agent-scratch/claude"},
		{"/c/agent-scratch/claude", "C:/agent-scratch/claude"},
		{"relative/not/a/root", ""},
		{"/agent-scratch/claude", ""}, // no drive component: nothing to alias
		{"", ""},
	} {
		got := scratchpadRootAlias(tc.in)
		if runtime.GOOS != "windows" {
			if got != "" {
				t.Errorf("scratchpadRootAlias(%q)=%q, want %q off Windows — a `C:`-prefixed alias would split a POSIX ':'-separated list into a bogus top-level root", tc.in, got, "")
			}
			continue
		}
		if got != tc.want {
			t.Errorf("scratchpadRootAlias(%q)=%q, want %q", tc.in, got, tc.want)
		}
	}
}
