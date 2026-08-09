package commitlane

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// TestClassifyStagedDeletionsMixed is the ticket's acceptance case (#5339): an index
// holding REAL staged deletions next to no-op `git rm --cached` residue must yield a
// remedy scoped to the no-ops ONLY. A detector that flagged every staged deletion would
// be worse than none — acting on it would un-stage a peer's genuine deletion — so the
// non-churn classes are asserted individually, not just by count.
func TestClassifyStagedDeletionsMixed(t *testing.T) {
	const headBlob = "1111111111111111111111111111111111111111"
	const otherBlob = "2222222222222222222222222222222222222222"

	facts := []StagedDeletionFact{
		// on disk, byte-identical to HEAD: pure index churn
		{Path: "internal/a/noop.go", OnDisk: true, DiskHash: headBlob, HeadHash: headBlob},
		// genuinely gone from the working tree: somebody's real deletion
		{Path: "internal/b/real_delete.go", OnDisk: false, HeadHash: headBlob},
		// present but DIFFERENT from HEAD: not a no-op, and not ours to touch
		{Path: "internal/c/modified.go", OnDisk: true, DiskHash: otherBlob, HeadHash: headBlob},
		// HEAD blob unreadable: fail closed
		{Path: "internal/d/unreadable.go", OnDisk: true, DiskHash: otherBlob},
		// a second no-op, to prove the remedy carries the whole churn set
		{Path: "internal/e/noop_two.go", OnDisk: true, DiskHash: otherBlob, HeadHash: otherBlob},
	}

	audit := ClassifyStagedDeletions(facts)

	want := map[string]StagedDeletionClass{
		"internal/a/noop.go":        StagedDeletionNoOpChurn,
		"internal/b/real_delete.go": StagedDeletionRealDelete,
		"internal/c/modified.go":    StagedDeletionDiskDiffers,
		"internal/d/unreadable.go":  StagedDeletionUnknown,
		"internal/e/noop_two.go":    StagedDeletionNoOpChurn,
	}
	got := map[string]StagedDeletionClass{}
	for _, row := range audit.Rows {
		got[row.Path] = row.Class
	}
	for path, wantClass := range want {
		t.Run(string(wantClass)+"/"+path, func(t *testing.T) {
			if got[path] != wantClass {
				t.Fatalf("class for %s = %q, want %q", path, got[path], wantClass)
			}
		})
	}

	t.Run("remedy_clears_only_the_noops", func(t *testing.T) {
		wantPaths := []string{"internal/a/noop.go", "internal/e/noop_two.go"}
		if strings.Join(audit.NoOpPaths, ",") != strings.Join(wantPaths, ",") {
			t.Fatalf("noop paths = %v, want %v", audit.NoOpPaths, wantPaths)
		}
		if audit.NoOpCount() != 2 {
			t.Fatalf("noop count = %d, want 2", audit.NoOpCount())
		}
		wantRemedy := "git restore --staged -- internal/a/noop.go internal/e/noop_two.go"
		if audit.Remedy != wantRemedy {
			t.Fatalf("remedy = %q, want %q", audit.Remedy, wantRemedy)
		}
		// The remedy must never reach a path the audit did not prove to be churn.
		for _, forbidden := range []string{"real_delete.go", "modified.go", "unreadable.go"} {
			if strings.Contains(audit.Remedy, forbidden) {
				t.Fatalf("remedy %q must not touch non-churn path %s", audit.Remedy, forbidden)
			}
		}
	})
}

func TestClassifyStagedDeletionsEmpty(t *testing.T) {
	audit := ClassifyStagedDeletions(nil)
	if len(audit.Rows) != 0 || audit.Remedy != "" || audit.NoOpCount() != 0 {
		t.Fatalf("empty facts should yield an empty audit, got %+v", audit)
	}
}

// TestClassifyStagedDeletionsErrorFactFailsClosed pins the fail-closed posture: a probe
// error on a path that would otherwise look byte-identical must NOT enter the remedy.
func TestClassifyStagedDeletionsErrorFactFailsClosed(t *testing.T) {
	const blob = "3333333333333333333333333333333333333333"
	audit := ClassifyStagedDeletions([]StagedDeletionFact{
		{Path: "x.go", OnDisk: true, DiskHash: blob, HeadHash: blob, Err: "hash probe failed"},
	})
	if len(audit.Rows) != 1 || audit.Rows[0].Class != StagedDeletionUnknown {
		t.Fatalf("errored fact = %+v, want a single unknown row", audit.Rows)
	}
	if audit.Remedy != "" || audit.NoOpCount() != 0 {
		t.Fatalf("errored fact must not be offered for clearing, got remedy %q", audit.Remedy)
	}
}

func TestRestoreStagedCommandQuotesAndScopes(t *testing.T) {
	if got := RestoreStagedCommand(nil); got != "" {
		t.Fatalf("no paths should render no command, got %q", got)
	}
	got := RestoreStagedCommand([]string{"a b/c.go", "d.go"})
	want := `git restore --staged -- "a b/c.go" d.go`
	if got != want {
		t.Fatalf("remedy = %q, want %q", got, want)
	}
}

// TestScanStagedDeletionsGathersMixedIndex drives the thin impure half through the same
// injected Runner seam the rest of the package uses, proving the git wiring (a NUL-split
// name-only diff, batched ls-tree, batched hash-object) lands the right facts on the pure
// classifier — including the argument-ordered hash-object pairing.
func TestScanStagedDeletionsGathersMixedIndex(t *testing.T) {
	const noopBlob = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	const goneBlob = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	const diskBlob = "cccccccccccccccccccccccccccccccccccccccc"
	root := t.TempDir()

	run := func(_ context.Context, _ string, args ...string) RunResult {
		args = stripNoOptional(args)
		joined := strings.Join(args, " ")
		switch {
		case joined == "diff --cached --diff-filter=D --name-only -z --":
			return RunResult{Stdout: "keep.go\x00gone.go\x00edited.go\x00"}
		case strings.HasPrefix(joined, "ls-tree -z HEAD --"):
			return RunResult{Stdout: "100644 blob " + noopBlob + "\tkeep.go\x00" +
				"100644 blob " + goneBlob + "\tgone.go\x00" +
				"100644 blob " + goneBlob + "\tedited.go\x00"}
		case joined == "hash-object -- keep.go edited.go":
			// One line per argument, in argument order.
			return RunResult{Stdout: noopBlob + "\n" + diskBlob + "\n"}
		}
		return RunResult{Code: 1, Stderr: "unexpected git args: " + joined}
	}
	stat := func(path string) FileFact {
		if filepath.Base(path) == "gone.go" {
			return FileFact{}
		}
		return FileFact{Exists: true}
	}

	audit := ScanStagedDeletions(context.Background(), run, stat, root)

	want := map[string]StagedDeletionClass{
		"keep.go":   StagedDeletionNoOpChurn,
		"gone.go":   StagedDeletionRealDelete,
		"edited.go": StagedDeletionDiskDiffers,
	}
	if len(audit.Rows) != len(want) {
		t.Fatalf("rows = %+v, want %d", audit.Rows, len(want))
	}
	for _, row := range audit.Rows {
		if want[row.Path] != row.Class {
			t.Fatalf("class for %s = %q, want %q", row.Path, row.Class, want[row.Path])
		}
	}
	if audit.Remedy != "git restore --staged -- keep.go" {
		t.Fatalf("remedy = %q, want the no-op path only", audit.Remedy)
	}
}

// TestScanStagedDeletionsCleanIndexCostsOneRead proves the clean-index path is cheap and
// silent: one name-only diff, no hash reads, and an empty audit.
func TestScanStagedDeletionsCleanIndexCostsOneRead(t *testing.T) {
	calls := 0
	run := func(_ context.Context, _ string, args ...string) RunResult {
		calls++
		if strings.Join(stripNoOptional(args), " ") != "diff --cached --diff-filter=D --name-only -z --" {
			t.Fatalf("clean index must not issue %v", args)
		}
		return RunResult{Stdout: ""}
	}
	audit := ScanStagedDeletions(context.Background(), run, func(string) FileFact { return FileFact{} }, t.TempDir())
	if len(audit.Rows) != 0 || audit.Remedy != "" {
		t.Fatalf("clean index audit = %+v, want empty", audit)
	}
	if calls != 1 {
		t.Fatalf("clean index issued %d git calls, want 1", calls)
	}
}

// TestScanStagedDeletionsFailsOpenSilently pins the posture Status depends on: an
// unreadable probe yields an EMPTY audit, never a half-populated one that could put an
// unproven path in front of an operator.
func TestScanStagedDeletionsFailsOpenSilently(t *testing.T) {
	run := func(_ context.Context, _ string, _ ...string) RunResult {
		return RunResult{Code: 128, Stderr: "not a git repository"}
	}
	audit := ScanStagedDeletions(context.Background(), run, func(string) FileFact { return FileFact{} }, t.TempDir())
	if len(audit.Rows) != 0 || audit.Remedy != "" {
		t.Fatalf("failed probe audit = %+v, want empty", audit)
	}
}

// TestScanStagedDeletionsMisalignedHashesFailClosed covers the pairing hazard directly: if
// hash-object returns fewer lines than it was given paths, zipping them up would label a
// real deletion as churn. The whole chunk must degrade to unknown instead.
func TestScanStagedDeletionsMisalignedHashesFailClosed(t *testing.T) {
	const blob = "dddddddddddddddddddddddddddddddddddddddd"
	run := func(_ context.Context, _ string, args ...string) RunResult {
		joined := strings.Join(stripNoOptional(args), " ")
		switch {
		case joined == "diff --cached --diff-filter=D --name-only -z --":
			return RunResult{Stdout: "one.go\x00two.go\x00"}
		case strings.HasPrefix(joined, "ls-tree -z HEAD --"):
			return RunResult{Stdout: "100644 blob " + blob + "\tone.go\x00100644 blob " + blob + "\ttwo.go\x00"}
		case strings.HasPrefix(joined, "hash-object --"):
			return RunResult{Stdout: blob + "\n"} // one hash for two paths
		}
		return RunResult{Code: 1}
	}
	audit := ScanStagedDeletions(context.Background(), run, func(string) FileFact { return FileFact{Exists: true} }, t.TempDir())
	if audit.NoOpCount() != 0 || audit.Remedy != "" {
		t.Fatalf("misaligned hashes must not produce a remedy, got %+v", audit)
	}
	for _, row := range audit.Rows {
		if row.Class != StagedDeletionUnknown {
			t.Fatalf("row %+v should be unknown under misaligned hashes", row)
		}
	}
}

// TestStatusAttachesIndexChurnAudit proves the audit reaches the lane Report — the verb's
// carrier — and that it does NOT change the lane verdict: index churn is a hygiene
// finding, and promoting it to "blocked" would wedge the lane it exists to keep readable.
func TestStatusAttachesIndexChurnAudit(t *testing.T) {
	const blob = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
	root, gitDir := testRepoPaths(t)
	base := fakeRepoRunner(root, gitDir)
	run := func(ctx context.Context, dir string, args ...string) RunResult {
		joined := strings.Join(stripNoOptional(args), " ")
		switch {
		case joined == "diff --cached --diff-filter=D --name-only -z --":
			return RunResult{Stdout: "churn.go\x00"}
		case strings.HasPrefix(joined, "ls-tree -z HEAD --"):
			return RunResult{Stdout: "100644 blob " + blob + "\tchurn.go\x00"}
		case strings.HasPrefix(joined, "hash-object --"):
			return RunResult{Stdout: blob + "\n"}
		}
		return base(ctx, dir, args...)
	}
	rep, err := Status(context.Background(), Options{
		Runner:      run,
		ProbeLock:   func(path string) safecommit.LockProbe { return safecommit.LockProbe{Path: path} },
		Stat:        func(path string) FileFact { return FileFact{Exists: filepath.Base(path) == "churn.go"} },
		ProcessList: func(context.Context) ([]Process, error) { return nil, nil },
		Now:         fixedNow,
	})
	if err != nil {
		t.Fatal(err)
	}
	if rep.IndexChurn == nil || rep.IndexChurn.NoOpCount() != 1 {
		t.Fatalf("index churn audit = %+v, want one no-op row", rep.IndexChurn)
	}
	if rep.IndexChurn.Remedy != "git restore --staged -- churn.go" {
		t.Fatalf("remedy = %q", rep.IndexChurn.Remedy)
	}
	if !rep.OK || rep.Verdict != VerdictClear {
		t.Fatalf("index churn must not change the lane verdict, got %s/%v", rep.Verdict, rep.OK)
	}
}
