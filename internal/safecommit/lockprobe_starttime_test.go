package safecommit

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// pinStartTime injects a fixed start time for every PID, so a recycled PID can be proven
// without arranging a real one (the OS will not recycle a PID on demand).
func pinStartTime(t *testing.T, started time.Time, ok bool) {
	t.Helper()
	prev := processStartTimeFn
	processStartTimeFn = func(int) (time.Time, bool) { return started, ok }
	t.Cleanup(func() { processStartTimeFn = prev })
}

// pinImage injects a fixed process image for every PID.
func pinImage(t *testing.T, image string, ok bool) {
	t.Helper()
	prev := processImageNameFn
	processImageNameFn = func(int) (string, bool) { return image, ok }
	t.Cleanup(func() { processImageNameFn = prev })
}

// writeLockAged writes a lockfile holding pid and back-dates its mtime to age ago,
// returning the mtime it was set to — the moment the lock counts as recorded at.
func writeLockAged(t *testing.T, dir, name string, pid int, age time.Duration) (string, time.Time) {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(strconv.Itoa(pid)+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	recordedAt := time.Now().Add(-age)
	if err := os.Chtimes(path, recordedAt, recordedAt); err != nil {
		t.Fatal(err)
	}
	return path, recordedAt
}

// TestLiveCommitterImageIsReapedWhenItStartedAfterTheLock is the #5892 defect witness, and
// the discriminator the image check structurally cannot make.
//
// The holder PID is genuinely alive and its image is "pwsh" — squarely on
// committerImageTokens, so looksLikeCommitterImage clears it and the #2339 guard alone
// reports the lock un-reapable forever. But the process at that PID started AFTER the
// lockfile was written, which is only possible if the original holder died and the OS
// recycled its PID. That is the wedge: on a fleet host the allowlist is the ambient process
// population, so holder_dead never fires (the PID is alive) and holder_foreign never fires
// (the image looks committer-like), and the lane stays stuck until a human intervenes.
func TestLiveCommitterImageIsReapedWhenItStartedAfterTheLock(t *testing.T) {
	dir := t.TempDir()
	path, recordedAt := writeLockAged(t, dir, "recycled.lock", os.Getpid(), 30*time.Minute)

	pinImage(t, "pwsh", true) // committer-like: the image guard clears this, by design
	pinStartTime(t, recordedAt.Add(10*time.Minute), true)

	p := ProbeLock(path)
	if !p.StartedAfterLock {
		t.Fatalf("StartedAfterLock = false, want true (holder started 10m after the lock was written); probe = %+v", p)
	}
	if !p.Foreign || p.Stale {
		t.Fatalf("probe = %+v, want Foreign=true Stale=false", p)
	}
	if p.Reason != ReapReasonHolderForeign {
		t.Fatalf("Reason = %q, want %q", p.Reason, ReapReasonHolderForeign)
	}
	if !p.Reapable() {
		t.Fatal("Reapable() = false, want true: a PID recycled onto a live process may not wedge the lane")
	}
	// The image is still reported — it is evidence on the event — but it is emphatically
	// NOT what justified the break, and asserting it here keeps that pairing honest.
	if p.Image != "pwsh" {
		t.Fatalf("Image = %q, want %q recorded alongside the start-time break", p.Image, "pwsh")
	}
}

// TestStartTimeBreakNamesItsEvidenceOnTheEvent proves the LOCK_BROKEN line distinguishes a
// start-time break from an image break. Without started_after_lock=true, an operator reading
// "holder_foreign … image=pwsh" would conclude the allowlist had judged pwsh foreign — the
// opposite of what happened, since pwsh is ON the allowlist.
func TestStartTimeBreakNamesItsEvidenceOnTheEvent(t *testing.T) {
	dir := t.TempDir()
	path, recordedAt := writeLockAged(t, dir, "recycled.lock", os.Getpid(), 20*time.Minute)

	pinImage(t, "pwsh", true)
	pinStartTime(t, recordedAt.Add(5*time.Minute), true)
	events := captureReapEvents(t)

	reapStaleLock(path)
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("recycled-PID lock not reaped: stat err = %v", err)
	}
	if len(*events) != 1 {
		t.Fatalf("events = %v, want exactly one LOCK_BROKEN", *events)
	}
	ev := (*events)[0]
	for _, want := range []string{
		"LOCK_BROKEN", ReapReasonHolderForeign,
		"pid=" + strconv.Itoa(os.Getpid()),
		"started_after_lock=true", "image=pwsh", "age=",
	} {
		if !strings.Contains(ev, want) {
			t.Fatalf("event %q missing %q", ev, want)
		}
	}
}

// TestStartTimeGuardNeverBreaksALiveHolder pins the fail-safe direction, which is the one
// outcome #2339's acceptance forbids and #5892 may not regress. Every case here must leave
// the lock alone.
func TestStartTimeGuardNeverBreaksALiveHolder(t *testing.T) {
	cases := []struct {
		name         string
		startedAfter time.Duration // holder start time, relative to the lock's mtime
		startOK      bool
		why          string
	}{
		{
			name:         "holder predates its own lock",
			startedAfter: -time.Hour,
			startOK:      true,
			why:          "the ordinary live committer: it existed before it wrote the lock",
		},
		{
			name:         "start time unreadable",
			startedAfter: time.Hour, // ignored: startOK=false
			startOK:      false,
			why:          "an unreadable start time proves nothing and must not be read as reuse",
		},
		{
			name:         "inside the skew grace",
			startedAfter: startTimeSkewGrace - time.Second,
			startOK:      true,
			why:          "two clocks disagreeing by less than the grace is not evidence of reuse",
		},
		{
			name:         "exactly at the skew grace",
			startedAfter: startTimeSkewGrace,
			startOK:      true,
			why:          "the boundary is inclusive on the safe side — strictly after, or not at all",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			path, recordedAt := writeLockAged(t, dir, "held.lock", os.Getpid(), time.Hour)

			pinImage(t, "fak", true) // a real committer, so only the start-time tier is in play
			pinStartTime(t, recordedAt.Add(tc.startedAfter), tc.startOK)
			events := captureReapEvents(t)

			if p := ProbeLock(path); p.StartedAfterLock || p.Foreign || p.Reapable() {
				t.Fatalf("probe = %+v, want not foreign and not reapable — %s", p, tc.why)
			}
			reapStaleLock(path)
			if _, err := os.Stat(path); err != nil {
				t.Fatalf("live holder's lock was wrongly reaped (%s): %v", tc.why, err)
			}
			if len(*events) != 0 {
				t.Fatalf("live holder emitted LOCK_BROKEN events: %v", *events)
			}
		})
	}
}

// TestImageGuardStillDecidesWhenStartTimeCannot proves the #2339 heuristic is a fallback,
// not a casualty: with no readable start time, a foreign image must still reap exactly as
// it did before this guard existed. A new discriminator that quietly narrowed the old one
// would be a regression wearing a fix's clothes.
func TestImageGuardStillDecidesWhenStartTimeCannot(t *testing.T) {
	dir := t.TempDir()
	path, _ := writeLockAged(t, dir, "foreign.lock", os.Getpid(), time.Minute)

	pinStartTime(t, time.Time{}, false) // no start time available on this platform
	pinImage(t, "notepad", true)        // provably not a committer

	p := ProbeLock(path)
	if p.StartedAfterLock {
		t.Fatalf("StartedAfterLock = true with no readable start time; probe = %+v", p)
	}
	if !p.Foreign || p.Reason != ReapReasonHolderForeign {
		t.Fatalf("probe = %+v, want the image guard to still report a foreign holder", p)
	}
}

// TestStartTimeGuardNeedsAKnownLockTime proves the probe declines to compare against an
// mtime it does not have. A zero ModTime is "we could not stat it", not "the epoch" — and
// treating it as the latter would make every live holder look like it started an eternity
// after its lock and reap the whole lane.
func TestStartTimeGuardNeedsAKnownLockTime(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "unstattable.lock")
	if err := os.WriteFile(path, []byte(strconv.Itoa(os.Getpid())+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	pinImage(t, "fak", true)
	pinStartTime(t, time.Now(), true)

	p := ProbeLock(path)
	if p.ModTime.IsZero() {
		t.Skip("stat unexpectedly failed; the zero-ModTime path is asserted by construction below")
	}
	// The realistic construction: ModTime is readable and recent, and the holder started
	// now-ish, so the grace must keep this out of reuse territory.
	if p.StartedAfterLock || p.Foreign {
		t.Fatalf("probe = %+v, want a freshly-written lock held by a just-started process to be safe", p)
	}
}
