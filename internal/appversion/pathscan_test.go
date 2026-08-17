package appversion

import (
	"path/filepath"
	"testing"
)

// probeMap builds a probe closure over a fixed set of existing paths (keyed by
// cleaned path) so ScanPathForFak can be exercised without touching the disk.
func probeMap(existing map[string]PathBinary) func(string) (PathBinary, bool) {
	return func(path string) (PathBinary, bool) {
		b, ok := existing[cleanPathKey(path)]
		return b, ok
	}
}

func TestScanPathForFakRanksInResolutionOrderAndDedups(t *testing.T) {
	binDir := filepath.Join("C:", "Users", "u", "bin")
	goDir := filepath.Join("C:", "Users", "u", "go", "bin")
	existing := map[string]PathBinary{
		cleanPathKey(filepath.Join(binDir, "fak.exe")): {Stamped: false},
		cleanPathKey(filepath.Join(goDir, "fak.exe")):  {Stamped: true, Commit: "abc", CommitTime: "2026-07-03T00:00:00Z"},
	}
	// PATH lists binDir twice (a common real duplication) then goDir.
	dirs := []string{binDir, binDir, goDir}
	got := ScanPathForFak(dirs, []string{"fak.exe", "fak"}, probeMap(existing))
	if len(got) != 2 {
		t.Fatalf("want 2 resolved binaries, got %d: %+v", len(got), got)
	}
	if !got[0].Winner || got[0].Rank != 0 {
		t.Errorf("first entry must be the winner at rank 0: %+v", got[0])
	}
	if got[0].Winner == got[1].Winner {
		t.Errorf("exactly one winner expected, got %+v / %+v", got[0], got[1])
	}
	if cleanPathKey(got[0].Path) != cleanPathKey(filepath.Join(binDir, "fak.exe")) {
		t.Errorf("winner should be the bin/ copy (first on PATH), got %s", got[0].Path)
	}
}

func TestPathShadowRecommendationWarnsOnUnstampedWinner(t *testing.T) {
	bins := []PathBinary{
		{Path: "/u/bin/fak", Winner: true, Stamped: false, ModTime: "2026-07-09T06:00:00Z"},
		{Path: "/u/go/bin/fak", Stamped: true, Commit: "deadbeef", CommitTime: "2026-07-03T00:00:00Z", ModTime: "2026-07-03T00:00:00Z"},
	}
	rec := PathShadowRecommendation(bins)
	if rec.Severity != SeverityWarn {
		t.Fatalf("unstamped winner must WARN, got %q (%s)", rec.Severity, rec.Finding)
	}
	if rec.Check != "binary-path-shadow" {
		t.Errorf("unexpected check name %q", rec.Check)
	}
}

func TestPathShadowRecommendationWarnsWhenWinnerOlderThanSibling(t *testing.T) {
	bins := []PathBinary{
		{Path: "/u/bin/fak", Winner: true, Stamped: true, Commit: "old", CommitTime: "2026-07-01T00:00:00Z"},
		{Path: "/u/go/bin/fak", Stamped: true, Commit: "new", CommitTime: "2026-07-08T00:00:00Z"},
	}
	rec := PathShadowRecommendation(bins)
	if rec.Severity != SeverityWarn {
		t.Fatalf("stale winner shadowing a newer sibling must WARN, got %q (%s)", rec.Severity, rec.Finding)
	}
}

func TestPathShadowRecommendationOKWhenWinnerStampedAndNewest(t *testing.T) {
	bins := []PathBinary{
		{Path: "/u/bin/fak", Winner: true, Stamped: true, Commit: "new", CommitTime: "2026-07-09T00:00:00Z"},
		{Path: "/u/go/bin/fak", Stamped: true, Commit: "old", CommitTime: "2026-07-03T00:00:00Z"},
	}
	rec := PathShadowRecommendation(bins)
	if rec.Severity != SeverityOK {
		t.Fatalf("stamped newest winner should be OK, got %q (%s)", rec.Severity, rec.Finding)
	}
}

func TestPathShadowRecommendationSingleBinaryOK(t *testing.T) {
	bins := []PathBinary{
		{Path: "/u/bin/fak", Winner: true, Stamped: true, Commit: "x", CommitTime: "2026-07-09T00:00:00Z"},
	}
	if rec := PathShadowRecommendation(bins); rec.Severity != SeverityOK {
		t.Fatalf("a lone stamped fak should be OK, got %q (%s)", rec.Severity, rec.Finding)
	}
}

func TestBinaryNewerDoesNotOrderEqualCleanCommitsByMtime(t *testing.T) {
	olderFile := PathBinary{Stamped: true, Commit: "abc123", ModTime: "2026-07-01T00:00:00Z"}
	newerFile := PathBinary{Stamped: true, Commit: "abc123", ModTime: "2026-07-09T00:00:00Z"}
	if binaryNewer(newerFile, olderFile) || binaryNewer(olderFile, newerFile) {
		t.Fatal("equal clean commit identities must not be ordered by file mtime")
	}

	dirty := newerFile
	dirty.Dirty = true
	if !binaryNewer(dirty, olderFile) {
		t.Fatal("dirty and clean copies must not be collapsed to equal identity")
	}
}

func TestBinaryNewerPrefersCommitTimeThenMtime(t *testing.T) {
	// Both stamped: commit time decides even if mtime disagrees.
	a := PathBinary{Stamped: true, CommitTime: "2026-07-08T00:00:00Z", ModTime: "2020-01-01T00:00:00Z"}
	b := PathBinary{Stamped: true, CommitTime: "2026-07-01T00:00:00Z", ModTime: "2026-07-09T00:00:00Z"}
	if !binaryNewer(a, b) {
		t.Errorf("commit time should rank a newer than b")
	}
	// Winner unstamped: fall back to mtime.
	c := PathBinary{Stamped: false, ModTime: "2026-07-09T00:00:00Z"}
	d := PathBinary{Stamped: true, CommitTime: "2026-07-01T00:00:00Z", ModTime: "2026-07-01T00:00:00Z"}
	if !binaryNewer(c, d) {
		t.Errorf("mtime fallback should rank c newer than d")
	}
	// Undecidable: no times → not newer.
	if binaryNewer(PathBinary{}, PathBinary{}) {
		t.Errorf("no ordering signal should not claim newer")
	}
}
