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
// journal.Row and runs the REAL journal verifiers over it — proving the
// witness contract (pillar 2) against real fak-guard traffic when it exists.
// When no real journal exists on the host (a clean CI runner, a fresh clone),
// the test SKIPS with an explicit, honest message naming the gap — it never
// fabricates a substitute and never silently passes with zero rows checked.
//
// WHICH verifier, and why there are two. journal.VerifyRows reads a file as ONE
// LINEAR chain. That is the correct and complete check for a journal with a
// SINGLE writer process, which is the shipped default: `fak guard` gives every
// session its own file (cmd/fak/guard_support.go's guardDefaultAuditPath ->
// .dispatch-runs/guard-audit/interactive-<pid>-<hash>.jsonl), and this test
// holds every such file to it with no allowance whatsoever.
//
// guardrsi.JournalPaths ALSO resolves the legacy shared <config>/fak/guard-audit.jsonl,
// which predates that per-session split and which many `fak guard` processes
// appended to at once. journal.Open recovers the chain head into PROCESS-LOCAL
// state, so concurrent writers each append from their own snapshot and the file
// becomes a hash TREE rather than a chain — a linear read then reports a
// "sequence gap" on a journal nobody edited. That is the case cmd/fak's `fak
// audit diagnose` already exists to classify (INTERLEAVED, not TAMPERED), so
// this test uses journal.VerifyForest for it rather than either waving the
// failure off or calling a benign interleave tampering.
//
// VerifyForest is a re-aim, NOT a relaxation: it runs the SAME journal.VerifyRows
// once per root-to-tip branch instead of once per file, and ADDS a requirement a
// linear read never makes at all — every non-genesis row's parent must be present,
// so a dropped or edited row is still refused. A file that verifies linearly is
// still required to; the fallback only ever applies where the linear read already
// failed, and a forked file that is genuinely tampered with still fails here.
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
	linearFiles := 0
	forkedFiles := 0
	forkedBranches := 0
	for _, path := range paths {
		rows, err := realJournalRows(path)
		if err != nil {
			t.Fatalf("parse real journal %s: %v", path, err)
		}
		if len(rows) == 0 {
			continue
		}
		n, linearErr := journal.VerifyRows(rows)
		if linearErr == nil {
			totalRows += n
			linearFiles++
			continue
		}
		// Not one linear chain. The ONLY reading of that a real capture is
		// allowed is "more than one process appended to this file", and even
		// then every branch must be cryptographically whole with no row
		// dropped, replayed, or edited. Anything else is tampering and fails.
		forest, forestErr := journal.VerifyForest(rows)
		if forestErr != nil {
			t.Fatalf("real captured journal %s is NOT an intact hash forest: %v\n"+
				"  (linear journal.VerifyRows stopped at row %d: %v; forest shape %+v)\n"+
				"  -- a real fak-guard chain must verify; this is not a synthetic fixture we can wave off, "+
				"and concurrent writers do not explain a missing parent, a replayed hash, or a broken branch",
				path, forestErr, n, linearErr, forest)
		}
		if forest.Orphans != 0 || forest.BrokenChains != 0 || forest.IntactChains != forest.Tips {
			t.Fatalf("real captured journal %s reported sound but its shape disagrees: %+v", path, forest)
		}
		totalRows += forest.Rows
		forkedFiles++
		forkedBranches += forest.Tips
		t.Logf("dogfood: %s is a concurrent-writer hash forest, INTACT: %d row(s), %d genesis, "+
			"%d branch point(s), %d branch(es) each verifying under journal.VerifyRows, 0 orphan/duplicate "+
			"(linear read stopped at row %d: %v)",
			path, forest.Rows, forest.Genesis, forest.BranchPoints, forest.IntactChains, n, linearErr)
	}
	if totalRows == 0 {
		t.Skipf("not yet: %d journal file(s) resolved but all were empty/blank on this host -- %s", len(paths), guardrsi.DiagnoseAuditGap(root))
	}
	if linearFiles == 0 && forkedFiles > 0 {
		t.Fatalf("not yet: %d real journal file(s) on this host and NOT ONE is a single-writer linear chain -- "+
			"the strict journal.VerifyRows contract went unexercised against real traffic. "+
			"Next checkable step: run `fak guard -- <agent>` once so a per-session "+
			".dispatch-runs/guard-audit/interactive-<pid>-*.jsonl exists, then re-run", forkedFiles)
	}
	t.Logf("dogfood: verified %d real row(s) across %d single-writer file(s) (strict linear journal.VerifyRows) "+
		"and %d concurrent-writer file(s) covering %d intact branch(es), out of %d resolved path(s)",
		totalRows, linearFiles, forkedFiles, forkedBranches, len(paths))
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
