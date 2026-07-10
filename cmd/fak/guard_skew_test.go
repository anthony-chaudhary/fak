package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/versionskew"
)

// TestGuardSkewBuildWarning pins the attested-but-stale banner warning. Skewed (provably behind)
// and Diverged (off-trunk) each warn with their distinct phrasing and both revs; every other
// verdict — including Unstamped, which guardUnattestedBuildWarning owns so the banner never
// double-warns — stays silent.
func TestGuardSkewBuildWarning(t *testing.T) {
	skewed := versionskew.Assessment{Verdict: versionskew.Skewed, Running: "aaaaaaaaaaaa1111", TrunkTip: "bbbbbbbbbbbb2222"}
	warn := guardSkewBuildWarning(skewed)
	for _, want := range []string{"provably BEHIND", "re-execs THIS file", "aaaaaaaaaaaa", "bbbbbbbbbbbb"} {
		if !strings.Contains(warn, want) {
			t.Fatalf("Skewed warning missing %q:\n%s", want, warn)
		}
	}
	if !strings.HasSuffix(warn, "\n") {
		t.Fatalf("Skewed warning must be a complete line: %q", warn)
	}

	div := guardSkewBuildWarning(versionskew.Assessment{Verdict: versionskew.Diverged, Running: "cccccccccccc3333", TrunkTip: "dddddddddddd4444"})
	if !strings.Contains(div, "OFF the trunk line") {
		t.Fatalf("Diverged warning should name the off-trunk case:\n%s", div)
	}

	for _, v := range []versionskew.Verdict{versionskew.Fresh, versionskew.Ahead, versionskew.Dirty, versionskew.Unstamped, versionskew.Unknown} {
		if got := guardSkewBuildWarning(versionskew.Assessment{Verdict: v, Running: "x", TrunkTip: "y"}); got != "" {
			t.Fatalf("verdict %v must not warn (owned elsewhere or not stale), got: %q", v, got)
		}
	}
}

// TestGuardInfoSkewNote is the pane twin of TestGuardSkewBuildWarning: the same Skewed/Diverged
// verdicts produce a pane note, and every non-refusable verdict (plus Unstamped, owned by
// guardInfoStalenessNote) stays quiet so the pane shows at most one staleness line.
func TestGuardInfoSkewNote(t *testing.T) {
	note := guardInfoSkewNote(versionskew.Assessment{Verdict: versionskew.Skewed, Running: "aaaaaaaaaaaa1111", TrunkTip: "bbbbbbbbbbbb2222"})
	for _, want := range []string{"stale-build WARN", "provably BEHIND", "aaaaaaaaaaaa", "bbbbbbbbbbbb"} {
		if !strings.Contains(note, want) {
			t.Fatalf("Skewed pane note missing %q:\n%s", want, note)
		}
	}
	if strings.HasSuffix(note, "\n") {
		t.Fatalf("pane note is one row without its own newline (the caller adds it): %q", note)
	}
	if div := guardInfoSkewNote(versionskew.Assessment{Verdict: versionskew.Diverged, Running: "c", TrunkTip: "d"}); !strings.Contains(div, "OFF the trunk line") {
		t.Fatalf("Diverged pane note should name the off-trunk case:\n%s", div)
	}
	for _, v := range []versionskew.Verdict{versionskew.Fresh, versionskew.Ahead, versionskew.Dirty, versionskew.Unstamped, versionskew.Unknown} {
		if got := guardInfoSkewNote(versionskew.Assessment{Verdict: v}); got != "" {
			t.Fatalf("verdict %v must not add a pane note, got: %q", v, got)
		}
	}
}
