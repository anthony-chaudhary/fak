package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/stepbaton"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// toolprocReapFixture builds the guard's on-disk layout under a temp root: the
// toolproc journal at <root>/.fak/toolproc/journal.jsonl and the step-advice
// sidecars beside it at <root>/.fak. It returns the journal path and the sidecar
// dir so the stop-hook seam (which derives the sidecar dir from the journal's
// grandparent) resolves hermetically inside the temp tree.
func toolprocReapFixture(t *testing.T) (journal, adviceDir string) {
	t.Helper()
	root := t.TempDir()
	adviceDir = filepath.Join(root, ".fak")
	journal = filepath.Join(adviceDir, "toolproc", "journal.jsonl")
	if err := os.MkdirAll(filepath.Dir(journal), 0o755); err != nil {
		t.Fatalf("mkdir journal dir: %v", err)
	}
	return journal, adviceDir
}

func writeSeamAdvice(t *testing.T, dir, id string) string {
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

// TestToolprocStopReapsClosingSessionStepAdvice is the SessionEnd wiring witness:
// a stop firing for a session deletes that session's own stepadvice sidecar while
// a sibling live session's fresh file survives, and the hook returns no error.
func TestToolprocStopReapsClosingSessionStepAdvice(t *testing.T) {
	journal, adviceDir := toolprocReapFixture(t)
	closing := writeSeamAdvice(t, adviceDir, "closing-sess")
	sibling := writeSeamAdvice(t, adviceDir, "other-live-sess")

	if err := toolprocHookOnce(strings.NewReader(`{"session_id":"closing-sess"}`), "stop", journal, toolproc.HookEnvelope{}, 9_000); err != nil {
		t.Fatalf("stop hook: %v", err)
	}
	if _, err := os.Stat(closing); !os.IsNotExist(err) {
		t.Errorf("closing session's sidecar not reaped on stop (stat err=%v)", err)
	}
	if _, err := os.Stat(sibling); err != nil {
		t.Errorf("a sibling live session's fresh sidecar must survive, got stat err=%v", err)
	}
}

// TestToolprocStopSweepsStaleOrphanStepAdvice proves the crash-recovery half at
// the seam: a stop firing (even for an unrelated session) sweeps a stepadvice
// orphan older than the grace floor that no clean exit ever deleted.
func TestToolprocStopSweepsStaleOrphanStepAdvice(t *testing.T) {
	journal, adviceDir := toolprocReapFixture(t)
	orphan := writeSeamAdvice(t, adviceDir, "dead-crash-sess")
	old := time.Now().Add(-48 * time.Hour)
	if err := os.Chtimes(orphan, old, old); err != nil {
		t.Fatalf("age orphan: %v", err)
	}
	fresh := writeSeamAdvice(t, adviceDir, "some-live-sess")

	if err := toolprocHookOnce(strings.NewReader(`{"session_id":"unrelated-sess"}`), "stop", journal, toolproc.HookEnvelope{}, 9_000); err != nil {
		t.Fatalf("stop hook: %v", err)
	}
	if _, err := os.Stat(orphan); !os.IsNotExist(err) {
		t.Errorf("stale orphan not swept on stop (stat err=%v)", err)
	}
	if _, err := os.Stat(fresh); err != nil {
		t.Errorf("a fresh sidecar must be kept by the age sweep, got stat err=%v", err)
	}
}

// TestToolprocStopReapErrorNeverFailsHook is the best-effort contract at the seam:
// even when the closing session's sidecar is un-removable (a non-empty directory
// shaped like the sidecar name), the stop hook still exits 0 and still bounds the
// journal — a reap fault can never surface as a hook failure.
func TestToolprocStopReapErrorNeverFailsHook(t *testing.T) {
	journal, adviceDir := toolprocReapFixture(t)
	// stepadvice-<id>.json as a NON-EMPTY directory forces os.Remove to fail.
	blocked := stepbaton.Path(adviceDir, "bad-sess")
	if err := os.MkdirAll(filepath.Join(blocked, "child"), 0o700); err != nil {
		t.Fatalf("build blocked sidecar dir: %v", err)
	}
	if err := os.WriteFile(journal, []byte(""), 0o644); err != nil {
		t.Fatalf("seed journal: %v", err)
	}

	// The inner helper swallows the reap error, so a clean stop returns nil.
	if err := toolprocHookOnce(strings.NewReader(`{"session_id":"bad-sess"}`), "stop", journal, toolproc.HookEnvelope{}, 9_000); err != nil {
		t.Fatalf("reap error must not surface from the stop hook, got %v", err)
	}
	// The un-removable directory is left intact (skipped, not force-deleted).
	if _, err := os.Stat(blocked); err != nil {
		t.Errorf("blocked sidecar dir must be left intact, got stat err=%v", err)
	}
	// The whole hook path stays fail-open (exit 0), the harness contract.
	var errOut strings.Builder
	if rc := runToolprocHook(strings.NewReader(`{"session_id":"bad-sess"}`), &errOut, []string{"stop", "--journal", journal}); rc != 0 {
		t.Errorf("stop hook must exit 0 despite a reap error, got %d (stderr=%q)", rc, errOut.String())
	}
}

// TestStepAdviceDirFromJournal pins the grandparent derivation: the sidecar dir is
// the journal's grandparent (<root>/.fak beside <root>/.fak/toolproc), and an
// empty journal path yields "" so the reap no-ops.
func TestStepAdviceDirFromJournal(t *testing.T) {
	if got := stepAdviceDirFromJournal(filepath.Join("r", ".fak", "toolproc", "journal.jsonl")); got != filepath.Join("r", ".fak") {
		t.Errorf("dir = %q, want %q", got, filepath.Join("r", ".fak"))
	}
	if got := stepAdviceDirFromJournal("   "); got != "" {
		t.Errorf("empty journal path -> %q, want empty", got)
	}
}
