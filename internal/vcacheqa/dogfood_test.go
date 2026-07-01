package vcacheqa

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/guardrsi"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

// TestDogfood_RealGuardAuditJournalVerifies is the issue's explicit dogfood
// requirement: "the harness runs against a real session transcript fixture
// captured from our own fak guard traffic, not a synthetic one." No such
// transcript is committed to the repo (every checked-in gateway fixture
// self-labels itself synthetic, e.g. internal/gateway/testdata/
// guard-trace-e2e.json's own "_provenance" field) and this package
// deliberately does NOT add one, for two reasons documented here rather than
// worked around silently:
//
//  1. A real fak-guard hash-chained journal capture is operator-private,
//     per-host state under .dispatch-runs/guard-audit/*.jsonl (the same path
//     internal/guardrsi.JournalPaths already resolves for the guard-RSI
//     scorecard) — committing one into testdata would ship another operator's
//     session bytes into the public tree, which AGENTS.md's FILE_ADMISSION
//     guidance reserves for noisy, operator-local, non-reproducible artifacts.
//  2. Fabricating a "real-looking" journal fixture to stand in for one would
//     be dishonest under this repo's own Law A2 sibling — a synthetic fixture
//     dressed as real evidence is exactly the self-report this harness exists
//     to refuse.
//
// So this test reads whatever real journal(s) already exist on THIS host via
// guardrsi.JournalPaths (the exact same resolver the guard-RSI scorecard
// already dogfoods against) and, if any are present, re-parses each row as a
// journal.Row and runs the REAL journal.VerifyRows over it — proving the
// witness contract (pillar 2) against real fak-guard traffic when it exists.
// When no real journal exists on the host (a clean CI runner, a fresh clone),
// the test SKIPS with an explicit, honest message naming the gap — it never
// fabricates a substitute and never silently passes with zero rows checked.
func TestDogfood_RealGuardAuditJournalVerifies(t *testing.T) {
	root, err := repoRootForTest()
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}
	paths := guardrsi.JournalPaths(root, "")
	if len(paths) == 0 {
		t.Skipf("not yet: no real .dispatch-runs/guard-audit/*.jsonl capture on this host -- %s. "+
			"This dogfood test opportunistically verifies real fak-guard traffic when present; "+
			"it is honestly skipped (not faked, not failed) when this host has none. "+
			"Next checkable step: run `fak guard -- <agent>` on this host at least once, then re-run this test.",
			guardrsi.DiagnoseAuditGap(root))
	}

	totalRows := 0
	verifiedFiles := 0
	for _, path := range paths {
		rows, err := realJournalRows(path)
		if err != nil {
			t.Fatalf("parse real journal %s: %v", path, err)
		}
		if len(rows) == 0 {
			continue
		}
		n, err := journal.VerifyRows(rows)
		if err != nil {
			t.Fatalf("real captured journal %s FAILED VerifyRows at row %d: %v -- a real fak-guard chain must verify; this is not a synthetic fixture we can wave off", path, n, err)
		}
		totalRows += n
		verifiedFiles++
	}
	if totalRows == 0 {
		t.Skipf("not yet: %d journal file(s) resolved but all were empty/blank on this host -- %s", len(paths), guardrsi.DiagnoseAuditGap(root))
	}
	t.Logf("dogfood: verified %d real row(s) across %d real fak-guard journal file(s) (%s)", totalRows, verifiedFiles, strings.Join(paths, ", "))
}

// realJournalRows reads one JSONL file and parses each non-blank line as a
// journal.Row. A line that fails to parse as a Row is skipped (mirroring
// guardrsi.FoldRows's own tolerant per-line handling) rather than aborting the
// whole file, since some rows may carry only a subset of Row's fields; the
// count of rows actually chained is what journal.VerifyRows reports.
func realJournalRows(path string) ([]journal.Row, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var rows []journal.Row
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var row journal.Row
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		if row.Hash == "" {
			continue // not a chained row (defensive: skip anything malformed/partial)
		}
		rows = append(rows, row)
	}
	return rows, nil
}

// repoRootForTest walks up from the working directory to find the go.mod
// root, mirroring the convention internal/conflationscore's own tests use
// (Build("../..")) but resolved dynamically since this test's relative depth
// must stay correct if the package ever moves.
func repoRootForTest() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if _, err := os.Stat(dir + string(os.PathSeparator) + "go.mod"); err == nil {
			return dir, nil
		}
		parent := parentDir(dir)
		if parent == dir {
			return "", os.ErrNotExist
		}
		dir = parent
	}
}

func parentDir(dir string) string {
	i := strings.LastIndexAny(dir, `/\`)
	if i <= 0 {
		return dir
	}
	return dir[:i]
}
