package safecommit

// reapfailure_test.go — the observability half of #5335.
//
// A .git/fak-commit.lock holding a DEAD pid wedged the whole fleet's commit lane for 85
// minutes. acquireWithReap calls reapStaleLock every 250ms for the whole of each
// committer's bounded wait (DefaultLockTimeout, 10s), so every queued attempt across those
// 85 minutes ran the reaper ~40 times and its os.Remove must have been failing every time
// — yet ReapResult had nowhere to put the error and reapStaleLock returned early without
// emitting anything, so the incident produced ZERO diagnostic. These tests pin that a
// refused remove now surfaces its reason, and that the fail-safe reap POLICY (which locks
// may be broken at all) is unchanged.

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// withLockRemove swaps the injected remove for the duration of a test and restores it.
func withLockRemove(t *testing.T, fn func(string) error) {
	t.Helper()
	prev := lockRemove
	lockRemove = fn
	t.Cleanup(func() { lockRemove = prev })
}

// withReapEvents captures the LOCK_BROKEN / LOCK_REAP_FAILED event stream.
func withReapEvents(t *testing.T) *[]string {
	t.Helper()
	var got []string
	prev := reapEventf
	reapEventf = func(format string, args ...any) {
		got = append(got, fmt.Sprintf(format, args...))
	}
	t.Cleanup(func() { reapEventf = prev })
	return &got
}

// resetReapFailures clears the cross-test throttle state so each case starts on a fresh,
// immediately-announced first attempt.
func resetReapFailures(t *testing.T) {
	t.Helper()
	reapFailureMu.Lock()
	reapFailures = map[string]*reapFailureStreak{}
	reapFailureMu.Unlock()
	t.Cleanup(func() {
		reapFailureMu.Lock()
		reapFailures = map[string]*reapFailureStreak{}
		reapFailureMu.Unlock()
	})
}

// seedDeadHolderLock writes a lockfile recording a pid that is no longer running — the
// exact shape of the file that wedged the lane.
func seedDeadHolderLock(t *testing.T, dir string) string {
	t.Helper()
	path := filepath.Join(dir, "fak-commit.lock")
	if err := os.WriteFile(path, []byte(strconv.Itoa(deadPID(t))+"\n"), 0o644); err != nil {
		t.Fatalf("seed: %v", err)
	}
	return path
}

// TestReapStaleLockResultSurfacesRemoveFailure is the core Change-1 witness: when the
// probe judges a lock breakable and the remove is REFUSED, the outcome carries the error,
// its closed class, the decision Reason, and Attempted=true — instead of the bare
// Reaped=false that made an 85-minute wedge indistinguishable from a healthy "a live
// committer holds it".
func TestReapStaleLockResultSurfacesRemoveFailure(t *testing.T) {
	cases := []struct {
		name      string
		err       error
		wantClass string
	}{
		{
			name:      "windows sharing violation — a still-open handle blocks the delete",
			err:       &fs.PathError{Op: "remove", Err: syscall.Errno(winErrSharingViolation)},
			wantClass: reapBusyClassForGOOS(),
		},
		{
			name:      "access denied — the classic Windows refusal on a held lockfile",
			err:       &fs.PathError{Op: "remove", Err: fs.ErrPermission},
			wantClass: ReapRemoveErrPermission,
		},
		{
			name:      "raced away between the probe and the remove",
			err:       &fs.PathError{Op: "remove", Err: fs.ErrNotExist},
			wantClass: ReapRemoveErrNotExist,
		},
		{
			name:      "anything else is still recorded, not swallowed",
			err:       errors.New("filesystem went away"),
			wantClass: ReapRemoveErrOther,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := seedDeadHolderLock(t, t.TempDir())
			withLockRemove(t, func(string) error { return tc.err })

			res := ReapStaleLockResult(path)

			if res.Reaped {
				t.Fatalf("remove failed but result claims Reaped: %+v", res)
			}
			if !res.Attempted {
				t.Fatalf("a decided-then-refused reap must record Attempted=true: %+v", res)
			}
			if !res.Failed() {
				t.Fatalf("Failed() must distinguish a refused remove from an untouched live lock: %+v", res)
			}
			if res.RemoveErr == "" {
				t.Fatalf("the os.Remove error was discarded — the exact #5335 defect: %+v", res)
			}
			if res.RemoveErrClass != tc.wantClass {
				t.Errorf("RemoveErrClass = %q, want %q (err %v)", res.RemoveErrClass, tc.wantClass, tc.err)
			}
			if res.Reason != ReapReasonHolderDead {
				t.Errorf("Reason = %q, want %q — the break DECISION is still explainable after a failed remove",
					res.Reason, ReapReasonHolderDead)
			}
			if res.HolderPID <= 0 {
				t.Errorf("HolderPID = %d, want the dead holder recorded in the file", res.HolderPID)
			}
		})
	}
}

// reapBusyClassForGOOS names the class a bare ERROR_SHARING_VIOLATION errno must map to on
// this platform. The Win32 sharing/lock-violation codes are only interpreted as such on
// Windows — on unix those same numbers mean something unrelated (EPIPE/EDOM), so the
// classifier deliberately declines to read them there. The expectation is derived from
// runtime.GOOS, NOT by calling classifyRemoveErr: asking the function under test what it
// expects would make the sharing-violation case a tautology that survives deleting the
// Windows branch outright.
func reapBusyClassForGOOS() string {
	if runtime.GOOS == "windows" {
		return ReapRemoveErrBusy
	}
	return ReapRemoveErrOther
}

// TestReapStaleLockResultKeepsSuccessAndFailSafePaths pins that Change 1 is OBSERVABILITY
// ONLY: the set of locks that get broken is exactly what it was. A dead holder still
// reaps (with no failure fields set), and a live holder / absent file / unattributable
// body are all still left untouched with Attempted=false — never mislabelled as failures.
func TestReapStaleLockResultKeepsSuccessAndFailSafePaths(t *testing.T) {
	t.Run("dead holder still reaps cleanly", func(t *testing.T) {
		path := seedDeadHolderLock(t, t.TempDir())
		res := ReapStaleLockResult(path)
		if !res.Reaped || res.Reason != ReapReasonHolderDead {
			t.Fatalf("dead-holder reap regressed: %+v", res)
		}
		if res.Failed() || res.RemoveErr != "" || res.RemoveErrClass != "" {
			t.Fatalf("a successful reap must set no failure fields: %+v", res)
		}
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("lock not removed: %v", err)
		}
	})

	dir := t.TempDir()
	notReapable := map[string]string{
		"live holder (this process)": strconv.Itoa(os.Getpid()) + "\n",
		"empty body":                 "",
		"non-numeric body":           "not-a-pid\n",
	}
	for name, body := range notReapable {
		t.Run(name+" is untouched and NOT a failure", func(t *testing.T) {
			path := filepath.Join(dir, "keep.lock")
			if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
				t.Fatalf("seed: %v", err)
			}
			withLockRemove(t, func(string) error {
				t.Fatal("remove must never be issued for a non-reapable lock")
				return nil
			})
			res := ReapStaleLockResult(path)
			if res.Reaped || res.Attempted || res.Failed() {
				t.Fatalf("non-reapable lock produced %+v, want an untouched no-op", res)
			}
			if res.Reason != "" {
				t.Errorf("Reason = %q, want \"\" for a lock that was never judged breakable", res.Reason)
			}
		})
	}

	t.Run("absent file is a no-op", func(t *testing.T) {
		res := ReapStaleLockResult(filepath.Join(dir, "absent.lock"))
		if res.Reaped || res.Attempted || res.Failed() {
			t.Fatalf("absent lock produced %+v, want an untouched no-op", res)
		}
	})
}

// TestReapStaleLockEmitsThrottledFailureEvent is the end-to-end witness on the EXISTING
// event path: a repeatedly-refused reap is announced immediately, then re-announced on a
// bounded cadence with a cumulative attempt count, so the 250ms re-reap loop leaves a
// readable trail rather than either silence or twenty thousand duplicate lines.
func TestReapStaleLockEmitsThrottledFailureEvent(t *testing.T) {
	resetReapFailures(t)
	events := withReapEvents(t)
	path := seedDeadHolderLock(t, t.TempDir())
	withLockRemove(t, func(string) error {
		return &fs.PathError{Op: "remove", Path: path, Err: fs.ErrPermission}
	})

	// The 250ms re-reap loop, compressed: many attempts inside one notice interval.
	for i := 0; i < 25; i++ {
		reapStaleLock(path)
	}
	if len(*events) != 1 {
		t.Fatalf("25 refused reaps inside one notice interval emitted %d events, want exactly 1:\n%s",
			len(*events), strings.Join(*events, "\n"))
	}
	first := (*events)[0]
	for _, want := range []string{
		"LOCK_REAP_FAILED",
		ReapReasonHolderDead,
		"attempts=1",
		"err=" + ReapRemoveErrPermission,
		path,
	} {
		if !strings.Contains(first, want) {
			t.Errorf("failure event %q missing %q", first, want)
		}
	}

	// Age the throttle past the notice interval: the streak re-announces, carrying the
	// cumulative count that makes a persistent wedge legible.
	reapFailureMu.Lock()
	reapFailures[path].lastNotice = time.Now().Add(-2 * reapFailureNoticeInterval)
	reapFailureMu.Unlock()

	reapStaleLock(path)
	if len(*events) != 2 {
		t.Fatalf("aged throttle did not re-announce: %v", *events)
	}
	if !strings.Contains((*events)[1], "attempts=26") {
		t.Errorf("re-announcement must carry the cumulative count, got %q", (*events)[1])
	}
}

// TestReapStaleLockKeepsSilentWhenNothingWasAttempted pins that the new event is scoped
// to genuine failures: a live holder's lock is left alone every 250ms by design, and that
// expected no-op must not become log noise.
func TestReapStaleLockKeepsSilentWhenNothingWasAttempted(t *testing.T) {
	resetReapFailures(t)
	events := withReapEvents(t)
	path := filepath.Join(t.TempDir(), "fak-commit.lock")
	if err := os.WriteFile(path, []byte(strconv.Itoa(os.Getpid())+"\n"), 0o644); err != nil {
		t.Fatalf("seed: %v", err)
	}

	for i := 0; i < 10; i++ {
		reapStaleLock(path)
	}
	if len(*events) != 0 {
		t.Fatalf("a live holder's untouched lock emitted events: %v", *events)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("live holder's lock was wrongly removed: %v", err)
	}
}

// TestReapFailureStreakResetsAfterASuccessfulReap pins that a resolved incident does not
// leave a throttle behind: the next failure streak announces its first attempt at once
// rather than waiting out a stale notice interval.
func TestReapFailureStreakResetsAfterASuccessfulReap(t *testing.T) {
	resetReapFailures(t)
	events := withReapEvents(t)
	dir := t.TempDir()
	path := seedDeadHolderLock(t, dir)

	withLockRemove(t, func(string) error { return &fs.PathError{Op: "remove", Err: fs.ErrPermission} })
	reapStaleLock(path)
	reapStaleLock(path)
	if len(*events) != 1 {
		t.Fatalf("want 1 throttled failure event, got %v", *events)
	}

	// The handle clears and the lock is finally broken.
	withLockRemove(t, os.Remove)
	reapStaleLock(path)
	if len(*events) != 2 || !strings.Contains((*events)[1], "LOCK_BROKEN") {
		t.Fatalf("successful break did not emit LOCK_BROKEN: %v", *events)
	}
	reapFailureMu.Lock()
	_, tracked := reapFailures[path]
	reapFailureMu.Unlock()
	if tracked {
		t.Fatalf("a successful reap must clear the failure streak for %s", path)
	}

	// A NEW streak on the same path announces immediately (attempts=1), not throttled.
	seedDeadHolderLock(t, dir)
	withLockRemove(t, func(string) error { return &fs.PathError{Op: "remove", Err: fs.ErrPermission} })
	reapStaleLock(path)
	if len(*events) != 3 || !strings.Contains((*events)[2], "attempts=1") {
		t.Fatalf("new failure streak did not announce from attempts=1: %v", *events)
	}
}
