package stepbatoncapture

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/stepbaton"
)

// writeAdvice writes a minimal per-session sidecar at the exact path
// stepbaton.Path produces for id, so these tests exercise the real name shape the
// reaper globs and deletes.
func writeAdvice(t *testing.T, dir, id string) string {
	t.Helper()
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
	p := stepbaton.Path(dir, id)
	if err := os.WriteFile(p, []byte("{}\n"), 0o600); err != nil {
		t.Fatalf("write %s: %v", p, err)
	}
	return p
}

// TestReapClosedAdviceDeletesClosingSession is the clean-exit witness: the closing
// trace's own file is deleted, a sibling live trace's file is untouched.
func TestReapClosedAdviceDeletesClosingSession(t *testing.T) {
	dir := t.TempDir()
	closing := writeAdvice(t, dir, "closing-sess")
	sibling := writeAdvice(t, dir, "other-live-sess")

	if err := ReapClosedAdvice(dir, "closing-sess"); err != nil {
		t.Fatalf("ReapClosedAdvice: %v", err)
	}
	if _, err := os.Stat(closing); !os.IsNotExist(err) {
		t.Errorf("closing trace's sidecar not deleted (stat err=%v)", err)
	}
	if _, err := os.Stat(sibling); err != nil {
		t.Errorf("a sibling live trace's sidecar must survive, got stat err=%v", err)
	}
}

// TestReapClosedAdviceAbsentIsNoError: deleting a trace with no sidecar (a session
// that never wrote one) is a success, not an error — the best-effort contract.
func TestReapClosedAdviceAbsentIsNoError(t *testing.T) {
	if err := ReapClosedAdvice(t.TempDir(), "never-wrote"); err != nil {
		t.Fatalf("absent sidecar must be a no-op success, got %v", err)
	}
}

// TestReapClosedAdviceEmptyArgsNoop: an empty dir or id never deletes anything.
func TestReapClosedAdviceEmptyArgsNoop(t *testing.T) {
	dir := t.TempDir()
	keep := writeAdvice(t, dir, "keep")
	if err := ReapClosedAdvice("", "keep"); err != nil {
		t.Fatalf("empty dir: %v", err)
	}
	if err := ReapClosedAdvice(dir, "  "); err != nil {
		t.Fatalf("blank id: %v", err)
	}
	if _, err := os.Stat(keep); err != nil {
		t.Errorf("empty-arg calls must delete nothing, got stat err=%v", err)
	}
}

// TestReapClosedAdviceErrorSurfaces proves the best-effort contract's SURFACE half:
// an un-removable target (a non-empty directory shaped like the sidecar name) makes
// os.Remove fail, and ReapClosedAdvice returns that error for the caller to log
// rather than panicking or hiding it. The caller (the stop-hook seam) is what
// swallows it — proven separately in the cmd/fak seam test.
func TestReapClosedAdviceErrorSurfaces(t *testing.T) {
	dir := t.TempDir()
	// stepadvice-<id>.json as a NON-EMPTY directory: os.Remove refuses it cross-platform.
	blocked := stepbaton.Path(dir, "blocked")
	if err := os.MkdirAll(filepath.Join(blocked, "child"), 0o700); err != nil {
		t.Fatalf("build blocked dir: %v", err)
	}
	if err := ReapClosedAdvice(dir, "blocked"); err == nil {
		t.Fatal("an un-removable sidecar must surface a non-nil error to the caller")
	}
}

// TestSweepStaleAdviceKeepsFreshRemovesOrphan is the age-sweep core: a fresh
// current-turn file (mtime now) is KEPT, an orphan past the floor is reaped.
func TestSweepStaleAdviceKeepsFreshRemovesOrphan(t *testing.T) {
	dir := t.TempDir()
	fresh := writeAdvice(t, dir, "live-now")
	orphan := writeAdvice(t, dir, "dead-crash")
	old := time.Now().Add(-48 * time.Hour)
	if err := os.Chtimes(orphan, old, old); err != nil {
		t.Fatalf("age the orphan: %v", err)
	}

	n, err := SweepStaleAdvice(dir, 24*time.Hour, time.Now())
	if err != nil {
		t.Fatalf("SweepStaleAdvice: %v", err)
	}
	if n != 1 {
		t.Fatalf("removed %d, want exactly 1 (the orphan)", n)
	}
	if _, err := os.Stat(orphan); !os.IsNotExist(err) {
		t.Errorf("orphan past the floor not swept (stat err=%v)", err)
	}
	if _, err := os.Stat(fresh); err != nil {
		t.Errorf("a fresh current-turn file must be kept, got stat err=%v", err)
	}
}

// TestSweepStaleAdviceKeepsNonAdviceFiles: the sweep touches only
// stepadvice-*.json — a sibling ledger or unrelated file, even ancient, is safe.
func TestSweepStaleAdviceKeepsNonAdviceFiles(t *testing.T) {
	dir := t.TempDir()
	other := filepath.Join(dir, "trajctl.jsonl")
	if err := os.WriteFile(other, []byte("x\n"), 0o600); err != nil {
		t.Fatalf("write sibling: %v", err)
	}
	old := time.Now().Add(-72 * time.Hour)
	if err := os.Chtimes(other, old, old); err != nil {
		t.Fatalf("age sibling: %v", err)
	}
	n, err := SweepStaleAdvice(dir, 24*time.Hour, time.Now())
	if err != nil {
		t.Fatalf("SweepStaleAdvice: %v", err)
	}
	if n != 0 {
		t.Fatalf("removed %d non-sidecar files, want 0", n)
	}
	if _, err := os.Stat(other); err != nil {
		t.Errorf("an ancient non-sidecar file must be kept, got stat err=%v", err)
	}
}

// TestSweepStaleAdviceMissingDirNoError: sweeping a dir that does not exist yet
// (no session ever wrote) is a no-op success, not an error.
func TestSweepStaleAdviceMissingDirNoError(t *testing.T) {
	n, err := SweepStaleAdvice(filepath.Join(t.TempDir(), "nope"), 24*time.Hour, time.Now())
	if err != nil || n != 0 {
		t.Fatalf("missing dir must be (0, nil), got (%d, %v)", n, err)
	}
}

// TestSweepStaleAdviceZeroFloorFallsBack: a mis-wired floor <= 0 must not sweep a
// fresh live file — it falls back to DefaultStaleFloor instead of a zero window
// that would take everything.
func TestSweepStaleAdviceZeroFloorFallsBack(t *testing.T) {
	dir := t.TempDir()
	fresh := writeAdvice(t, dir, "live")
	n, err := SweepStaleAdvice(dir, 0, time.Now())
	if err != nil {
		t.Fatalf("SweepStaleAdvice: %v", err)
	}
	if n != 0 {
		t.Fatalf("zero floor swept %d fresh files, want 0 (must fall back to the default floor)", n)
	}
	if _, err := os.Stat(fresh); err != nil {
		t.Errorf("fresh file swept under a zero floor, got stat err=%v", err)
	}
}
