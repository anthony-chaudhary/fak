package hooks

import (
	"context"
	"strings"
	"testing"
)

// gate_duplication_test.go — unit tests for the DUPLICATION advisory gate. The gate reads the
// candidate's added block (d.AddedByFile), the tracked .go listing (d.run "ls-files"), and the
// sibling contents (d.FileBytes / fileCache), so these drive it over a hand-built StagedDiff with
// a canned listing and pre-seeded file bytes — no real git, no disk.

// dupBlock is a block with enough logic over ~6 lines to yield at least one clonescan window
// (>34 normalized tokens carrying control/computation), so a genuine copy of it is detectable.
const dupBlock = `func accumulate(values []int, threshold int) int {
	total := 0
	for _, v := range values {
		if v > threshold {
			total += v
			continue
		}
		total -= v
	}
	return total
}
`

// addedLinesOf turns a source block into the AddedLine slice a staged diff would carry (added
// lines only, 1-based new-file line numbers).
func addedLinesOf(src string) []AddedLine {
	var out []AddedLine
	for i, ln := range strings.Split(strings.TrimRight(src, "\n"), "\n") {
		out = append(out, AddedLine{New: i + 1, Text: ln})
	}
	return out
}

// stagedForDup builds a StagedDiff whose added blocks, tracked .go listing, and sibling contents
// are all injected: added maps rel -> its added block, siblings maps rel -> committed source
// (pre-seeded so FileBytes never touches disk), and listing is what `git ls-files *.go` returns.
func stagedForDup(added, siblings map[string]string, listing []string) *StagedDiff {
	abf := map[string][]AddedLine{}
	for rel, src := range added {
		abf[rel] = addedLinesOf(src)
	}
	fc := map[string]fileEntry{}
	for rel, src := range siblings {
		fc[rel] = fileEntry{data: []byte(src), exists: true}
	}
	return &StagedDiff{
		Root:        ".",
		ctx:         context.Background(),
		AddedByFile: abf,
		fileCache:   fc,
		run: func(ctx context.Context, dir string, args ...string) (string, int, error) {
			return strings.Join(listing, "\n"), 0, nil
		},
	}
}

func siblingFile(block string) string { return "package foo\n\n" + block }

// A newly added block that copies a sibling in the SAME directory fires exactly one advisory
// finding naming the sibling site and the fix.
func TestGateDuplication_FiresOnIntraPackageClone(t *testing.T) {
	d := stagedForDup(
		map[string]string{"internal/foo/new.go": siblingFile(dupBlock)},
		map[string]string{"internal/foo/existing.go": siblingFile(dupBlock)},
		[]string{"internal/foo/existing.go"},
	)
	findings, err := gateDuplication(d)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("a copied block should fire exactly one finding, got %d: %+v", len(findings), findings)
	}
	f := findings[0]
	if f.Gate != "DUPLICATION" {
		t.Errorf("gate = %q, want DUPLICATION", f.Gate)
	}
	if f.File != "internal/foo/new.go" {
		t.Errorf("file = %q, want internal/foo/new.go", f.File)
	}
	for _, want := range []string{"internal/foo/existing.go", "FLEET_DUP_GUARD=block", "ALLOW_DUP=1"} {
		if !strings.Contains(f.Detail, want) {
			t.Errorf("detail %q missing %q", f.Detail, want)
		}
	}
}

// A distinct block with no token-similar sibling is clean.
func TestGateDuplication_CleanWhenNoClone(t *testing.T) {
	other := "func greet(name string) string {\n\treturn \"hello \" + name\n}\n"
	d := stagedForDup(
		map[string]string{"internal/foo/new.go": siblingFile(dupBlock)},
		map[string]string{"internal/foo/other.go": siblingFile(other)},
		[]string{"internal/foo/other.go"},
	)
	if findings, err := gateDuplication(d); err != nil || len(findings) != 0 {
		t.Fatalf("no token-similar sibling should be clean; got findings=%+v err=%v", findings, err)
	}
}

// The comparison is scoped to the candidate's OWN directory: an identical block that exists only
// in a DIFFERENT package is not read and does not fire (that clone is the whole-tree scorecard's
// job, not this gate's).
func TestGateDuplication_DifferentDirectoryOutOfScope(t *testing.T) {
	d := stagedForDup(
		map[string]string{"internal/foo/new.go": siblingFile(dupBlock)},
		map[string]string{"internal/bar/dup.go": siblingFile(dupBlock)},
		[]string{"internal/bar/dup.go"},
	)
	if findings, err := gateDuplication(d); err != nil || len(findings) != 0 {
		t.Fatalf("a clone in another package is out of scope; got findings=%+v err=%v", findings, err)
	}
}

// Test files are out of scope on both sides: a staged _test.go candidate never fires, and a
// _test.go sibling is not scanned.
func TestGateDuplication_TestFilesExcluded(t *testing.T) {
	// Candidate is a test file -> not a candidate at all.
	d := stagedForDup(
		map[string]string{"internal/foo/new_test.go": siblingFile(dupBlock)},
		map[string]string{"internal/foo/existing.go": siblingFile(dupBlock)},
		[]string{"internal/foo/existing.go"},
	)
	if findings, err := gateDuplication(d); err != nil || len(findings) != 0 {
		t.Fatalf("a _test.go candidate is out of scope; got findings=%+v err=%v", findings, err)
	}
	// Non-test candidate, but the only sibling holding the clone is a _test.go -> not scanned.
	d = stagedForDup(
		map[string]string{"internal/foo/new.go": siblingFile(dupBlock)},
		map[string]string{"internal/foo/existing_test.go": siblingFile(dupBlock)},
		[]string{"internal/foo/existing_test.go"},
	)
	if findings, err := gateDuplication(d); err != nil || len(findings) != 0 {
		t.Fatalf("a _test.go sibling must not be scanned; got findings=%+v err=%v", findings, err)
	}
}

// A block is never reported as a clone of ITSELF: a tracked file that appears in its own
// neighborhood is excluded via selfPath, so re-staging its own lines is clean when no OTHER
// sibling holds the block.
func TestGateDuplication_SelfNotReportedAsClone(t *testing.T) {
	d := stagedForDup(
		map[string]string{"internal/foo/x.go": siblingFile(dupBlock)},
		map[string]string{"internal/foo/x.go": siblingFile(dupBlock)},
		[]string{"internal/foo/x.go"},
	)
	if findings, err := gateDuplication(d); err != nil || len(findings) != 0 {
		t.Fatalf("a file must not clone itself; got findings=%+v err=%v", findings, err)
	}
}

// A directory over the neighborhood size cap is skipped (left to the whole-tree scorecard) so the
// per-commit cost stays bounded. Driven through the core with a cap of 1 against a 2-sibling dir.
func TestGateDuplication_OversizedNeighborhoodSkipped(t *testing.T) {
	d := stagedForDup(
		map[string]string{"internal/foo/new.go": siblingFile(dupBlock)},
		map[string]string{
			"internal/foo/a.go": siblingFile(dupBlock),
			"internal/foo/b.go": "package foo\n",
		},
		[]string{"internal/foo/a.go", "internal/foo/b.go"},
	)
	// cap 1 < 2 tracked siblings -> the directory is not read -> no findings.
	if findings, err := duplicationFindings(d, 1, 3); err != nil || len(findings) != 0 {
		t.Fatalf("an over-cap directory must be skipped; got findings=%+v err=%v", findings, err)
	}
	// A generous cap admits the same directory and the clone fires -> proves the cap is what
	// suppressed it above, not the fixture.
	if findings, err := duplicationFindings(d, 2500, 3); err != nil || len(findings) != 1 {
		t.Fatalf("under the cap the clone should fire; got findings=%+v err=%v", findings, err)
	}
}

// If the tracked listing cannot be read the gate fails OPEN (ErrCouldNotRun), never a false
// DUPLICATION — an advisory hygiene check must not wedge a commit when git is unreachable.
func TestGateDuplication_FailsOpenWhenListingUnavailable(t *testing.T) {
	d := stagedForDup(
		map[string]string{"internal/foo/new.go": siblingFile(dupBlock)},
		nil, nil,
	)
	d.run = func(ctx context.Context, dir string, args ...string) (string, int, error) {
		return "", 1, nil // git present, non-zero exit
	}
	if _, err := gateDuplication(d); err != ErrCouldNotRun {
		t.Errorf("a failed listing should fail open; err = %v, want ErrCouldNotRun", err)
	}
	// A nil runner (no git wired) is also could-not-run, not a panic.
	d.run = nil
	if _, err := gateDuplication(d); err != ErrCouldNotRun {
		t.Errorf("a nil runner should fail open; err = %v, want ErrCouldNotRun", err)
	}
}

// A commit whose only added .go block is below clonescan's qualifying window (a lone `package x`,
// a one-line tweak) fires no finding AND never spends the `git ls-files` subprocess: with no
// qualifying candidate there is no neighborhood worth reading, so trackedGoByDir is short-circuited
// entirely. The recording runner witnesses that `ls-files` was never invoked (#4327).
func TestGateDuplication_TrivialCandidateSkipsGitListing(t *testing.T) {
	var lsFilesCalls int
	d := &StagedDiff{
		Root:        ".",
		ctx:         context.Background(),
		AddedByFile: map[string][]AddedLine{"internal/foo/tiny.go": addedLinesOf("package foo\n\nvar enabled = true\n")},
		run: func(ctx context.Context, dir string, args ...string) (string, int, error) {
			for _, a := range args {
				if a == "ls-files" {
					lsFilesCalls++
				}
			}
			return "internal/foo/existing.go", 0, nil
		},
	}
	findings, err := gateDuplication(d)
	if err != nil {
		t.Fatalf("a trivial block must not error; err=%v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("a sub-window block cannot clone; got findings=%+v", findings)
	}
	if lsFilesCalls != 0 {
		t.Errorf("trackedGoByDir ran `git ls-files` %d time(s) for a commit with no qualifying candidate; the short-circuit must skip it (#4327)", lsFilesCalls)
	}
}

// Pruning is surgical: a trivial candidate staged alongside a qualifying one drops ONLY the trivial
// block. The qualifying candidate keeps its directory in the `touched` set, still reads its
// neighborhood, and fires the exact finding it did before the short-circuit — proving the narrowing
// never discards a provably-nonempty want-set (#4327).
func TestGateDuplication_TrivialPruneKeepsQualifyingCandidate(t *testing.T) {
	d := stagedForDup(
		map[string]string{
			"internal/foo/new.go":  siblingFile(dupBlock),         // qualifying -> must still fire
			"internal/foo/tiny.go": "package foo\n\nvar ok = 1\n", // trivial -> pruned before git read
		},
		map[string]string{"internal/foo/existing.go": siblingFile(dupBlock)},
		[]string{"internal/foo/existing.go"},
	)
	findings, err := gateDuplication(d)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("only the qualifying candidate should fire; got %d: %+v", len(findings), findings)
	}
	if findings[0].File != "internal/foo/new.go" {
		t.Errorf("finding file = %q, want internal/foo/new.go", findings[0].File)
	}
}
