package safecommit

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// captureReapEvents swaps reapEventf for a recorder and restores it on cleanup, so a
// test can assert the LOCK_BROKEN event without writing to real stderr.
func captureReapEvents(t *testing.T) *[]string {
	t.Helper()
	var lines []string
	prev := reapEventf
	reapEventf = func(format string, args ...any) {
		lines = append(lines, fmt.Sprintf(format, args...))
	}
	t.Cleanup(func() { reapEventf = prev })
	return &lines
}

// TestLockProbeAgeSeconds proves the probe reports the lock's age from its mtime — the
// "age=S" field the LOCK_BROKEN event carries so an operator sees how long the lane was
// wedged.
func TestLockProbeAgeSeconds(t *testing.T) {
	fixed := time.Date(2026, 7, 2, 12, 0, 0, 0, time.UTC)
	prev := nowFn
	nowFn = func() time.Time { return fixed }
	t.Cleanup(func() { nowFn = prev })

	path := filepath.Join(t.TempDir(), "aged.lock")
	if err := os.WriteFile(path, []byte(strconv.Itoa(os.Getpid())+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	old := fixed.Add(-90 * time.Second)
	if err := os.Chtimes(path, old, old); err != nil {
		t.Fatal(err)
	}
	if p := ProbeLock(path); p.AgeSeconds != 90 {
		t.Fatalf("AgeSeconds = %d, want 90 (mtime backdated 90s)", p.AgeSeconds)
	}
}

// TestForeignLivePIDIsReapedWithEvent proves the PID-reuse guard: a lock whose recorded
// PID is ALIVE but whose process image is provably not a committer (a reused PID number)
// is broken, and the break emits a holder_foreign LOCK_BROKEN event naming the image.
func TestForeignLivePIDIsReapedWithEvent(t *testing.T) {
	prev := processImageNameFn
	processImageNameFn = func(pid int) (string, bool) { return "notepad", true }
	t.Cleanup(func() { processImageNameFn = prev })

	events := captureReapEvents(t)
	path := filepath.Join(t.TempDir(), "reused.lock")
	// os.Getpid() is a genuinely-live PID; the injected image makes it look foreign.
	if err := os.WriteFile(path, []byte(strconv.Itoa(os.Getpid())+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	if p := ProbeLock(path); !p.Foreign || p.Stale || p.Reason != ReapReasonHolderForeign {
		t.Fatalf("probe = %+v, want Foreign=true Stale=false Reason=holder_foreign", p)
	}
	reapStaleLock(path)
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("foreign-live lock not reaped: stat err = %v", err)
	}
	if len(*events) != 1 {
		t.Fatalf("events = %v, want exactly one LOCK_BROKEN", *events)
	}
	ev := (*events)[0]
	for _, want := range []string{"LOCK_BROKEN", ReapReasonHolderForeign, "pid=" + strconv.Itoa(os.Getpid()), "image=notepad", "age="} {
		if !strings.Contains(ev, want) {
			t.Fatalf("event %q missing %q", ev, want)
		}
	}
}

// TestLiveCommitterImageNotReaped proves the safety gate the acceptance demands: a lock
// whose recorded PID is alive AND whose image looks like a committer (fak/git/go/…) is
// NEVER broken — reaping a live committer's lock would let two writers race the trunk.
func TestLiveCommitterImageNotReaped(t *testing.T) {
	prev := processImageNameFn
	processImageNameFn = func(pid int) (string, bool) { return "fak", true }
	t.Cleanup(func() { processImageNameFn = prev })

	events := captureReapEvents(t)
	path := filepath.Join(t.TempDir(), "committer.lock")
	if err := os.WriteFile(path, []byte(strconv.Itoa(os.Getpid())+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	reapStaleLock(path)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("live committer's lock was wrongly reaped: %v", err)
	}
	if len(*events) != 0 {
		t.Fatalf("live committer emitted LOCK_BROKEN events: %v", *events)
	}
}

// TestUnreadableImageLiveHolderNotReaped proves the fail-safe read direction: when the
// live holder's image CANNOT be read, the guard treats it as committer-like and does not
// break the lock — an unidentifiable live holder is never reaped.
func TestUnreadableImageLiveHolderNotReaped(t *testing.T) {
	prev := processImageNameFn
	processImageNameFn = func(pid int) (string, bool) { return "", false }
	t.Cleanup(func() { processImageNameFn = prev })

	path := filepath.Join(t.TempDir(), "unknown.lock")
	if err := os.WriteFile(path, []byte(strconv.Itoa(os.Getpid())+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if p := ProbeLock(path); p.Foreign || p.Reapable() {
		t.Fatalf("unreadable-image live holder classified reapable: %+v", p)
	}
	reapStaleLock(path)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("unidentifiable live holder's lock was wrongly reaped: %v", err)
	}
}

// TestDeadHolderEmitsHolderDeadEvent proves a dead-holder break emits the holder_dead
// LOCK_BROKEN event with the PID and age — the "logged event" the acceptance requires.
func TestDeadHolderEmitsHolderDeadEvent(t *testing.T) {
	events := captureReapEvents(t)
	path := filepath.Join(t.TempDir(), "dead.lock")
	pid := deadPID(t)
	if err := os.WriteFile(path, []byte(strconv.Itoa(pid)+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	reapStaleLock(path)
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("dead-holder lock not reaped: %v", err)
	}
	if len(*events) != 1 {
		t.Fatalf("events = %v, want exactly one LOCK_BROKEN", *events)
	}
	ev := (*events)[0]
	for _, want := range []string{"LOCK_BROKEN", ReapReasonHolderDead, "pid=" + strconv.Itoa(pid), "age="} {
		if !strings.Contains(ev, want) {
			t.Fatalf("event %q missing %q", ev, want)
		}
	}
}

// TestLooksLikeCommitterImage pins the classifier boundary the guard rests on.
func TestLooksLikeCommitterImage(t *testing.T) {
	committer := []string{"fak", "fak.exe", "git", "go", "safecommit.test", "pwsh", "powershell", "node", "bash", "", "  "}
	foreign := []string{"notepad", "chrome", "explorer", "python", "malware"}
	for _, name := range committer {
		if !looksLikeCommitterImage(name) {
			t.Errorf("looksLikeCommitterImage(%q) = false, want true (committer-like or unidentifiable)", name)
		}
	}
	for _, name := range foreign {
		if looksLikeCommitterImage(name) {
			t.Errorf("looksLikeCommitterImage(%q) = true, want false (foreign)", name)
		}
	}
}
