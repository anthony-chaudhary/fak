package metrics

import "testing"

// TestStaleLockWedgeIncident reproduces the 2026-07-13 KPI dispatch wedge: a
// crashed worker's `.git/index.lock` frozen at 20:16:31Z, its mtime unchanged
// across two samples, `.git/logs/HEAD` frozen at the 20:14Z last commit, and 8
// workers active. The classifier must land FROZEN_WEDGE, mark it safe to clear,
// and label the drought a frozen-lock wedge (not a batch lull).
func TestStaleLockWedgeIncident(t *testing.T) {
	// Epochs (Unix seconds) mirroring the incident timeline.
	const lockMod = 1_752_437_791 // 20:16:31Z, frozen across both samples
	const headLog = 1_752_437_640 // 20:14:00Z last commit, frozen
	in := WedgeInput{
		Prev:          LockSample{SampleUnix: lockMod + 20*60, LockPresent: true, LockModUnix: lockMod, HeadLogUnix: headLog},
		Curr:          LockSample{SampleUnix: lockMod + 26*60, LockPresent: true, LockModUnix: lockMod, HeadLogUnix: headLog},
		ActiveWorkers: 8,
		StaleAfter:    DefaultStaleLockWedgeAge,
	}
	v := ClassifyStaleLockWedge(in)
	if v.State != WedgeFrozen {
		t.Fatalf("state = %q, want %q (reason: %s)", v.State, WedgeFrozen, v.Reason)
	}
	if !v.SafeToClear {
		t.Fatalf("SafeToClear = false, want true for a confirmed frozen wedge")
	}
	if v.DroughtKind != DroughtFrozenLock {
		t.Fatalf("DroughtKind = %q, want %q", v.DroughtKind, DroughtFrozenLock)
	}
	if !v.Drought {
		t.Fatalf("Drought = false, want true (8 workers, no commits landing)")
	}
	if v.Event != EventLockRecoveredCandidate {
		t.Fatalf("Event = %q, want %q", v.Event, EventLockRecoveredCandidate)
	}
	if v.LockAgeSeconds != 26*60 {
		t.Fatalf("LockAgeSeconds = %d, want %d", v.LockAgeSeconds, 26*60)
	}
}

// TestStaleLockWedgeBatchLull is the benign case the watchdog must NOT touch: a
// live/churning lock (mtime advancing) with commits still landing (HEAD
// advancing). It must stay LIVE_LOCK, never safe to clear, with no drought.
func TestStaleLockWedgeBatchLull(t *testing.T) {
	const base = 1_752_400_000
	in := WedgeInput{
		Prev:          LockSample{SampleUnix: base + 60, LockPresent: true, LockModUnix: base + 10, HeadLogUnix: base},
		Curr:          LockSample{SampleUnix: base + 120, LockPresent: true, LockModUnix: base + 100, HeadLogUnix: base + 90},
		ActiveWorkers: 8,
		StaleAfter:    DefaultStaleLockWedgeAge,
	}
	v := ClassifyStaleLockWedge(in)
	if v.State != WedgeLive {
		t.Fatalf("state = %q, want %q (reason: %s)", v.State, WedgeLive, v.Reason)
	}
	if v.SafeToClear {
		t.Fatalf("SafeToClear = true, want false for a churning live lock")
	}
	if !v.LockAdvancing {
		t.Fatalf("LockAdvancing = false, want true")
	}
	if !v.HeadAdvancing {
		t.Fatalf("HeadAdvancing = false, want true")
	}
	if v.Drought {
		t.Fatalf("Drought = true, want false (commits are still landing)")
	}
	if v.DroughtKind != DroughtNone {
		t.Fatalf("DroughtKind = %q, want %q", v.DroughtKind, DroughtNone)
	}
}

// TestStaleLockWedgeFrozenLockButNoCommits is a lull where NO commits are landing
// (HEAD frozen) yet the lock is still churning (mtime advancing) — a live writer
// mid-commit. The window is a drought (workers active, no commits) but the cause
// is a live lock, so it is a batch-lull, never a frozen wedge.
func TestStaleLockWedgeFrozenLockButNoCommits(t *testing.T) {
	const base = 1_752_400_000
	in := WedgeInput{
		Prev:          LockSample{SampleUnix: base + 60, LockPresent: true, LockModUnix: base + 10, HeadLogUnix: base},
		Curr:          LockSample{SampleUnix: base + 120, LockPresent: true, LockModUnix: base + 100, HeadLogUnix: base},
		ActiveWorkers: 4,
		StaleAfter:    DefaultStaleLockWedgeAge,
	}
	v := ClassifyStaleLockWedge(in)
	if v.State != WedgeLive {
		t.Fatalf("state = %q, want %q (reason: %s)", v.State, WedgeLive, v.Reason)
	}
	if v.SafeToClear {
		t.Fatalf("SafeToClear = true, want false for a churning lock")
	}
	if !v.Drought {
		t.Fatalf("Drought = false, want true (workers active, no commits landing)")
	}
	if v.DroughtKind != DroughtBatchLull {
		t.Fatalf("DroughtKind = %q, want %q", v.DroughtKind, DroughtBatchLull)
	}
}

// TestStaleLockWedgeHeldLive: the full frozen signature holds but a recorded
// holder pid is alive — a live process owns the lock. Must veto the clear.
func TestStaleLockWedgeHeldLive(t *testing.T) {
	const lockMod = 1_752_437_791
	const headLog = 1_752_437_640
	in := WedgeInput{
		Prev:          LockSample{SampleUnix: lockMod + 20*60, LockPresent: true, LockModUnix: lockMod, HeadLogUnix: headLog},
		Curr:          LockSample{SampleUnix: lockMod + 26*60, LockPresent: true, LockModUnix: lockMod, HeadLogUnix: headLog},
		ActiveWorkers: 8,
		HolderKnown:   true,
		HolderPID:     4242,
		HolderAlive:   true,
		StaleAfter:    DefaultStaleLockWedgeAge,
	}
	v := ClassifyStaleLockWedge(in)
	if v.State != WedgeHeldLive {
		t.Fatalf("state = %q, want %q (reason: %s)", v.State, WedgeHeldLive, v.Reason)
	}
	if v.SafeToClear {
		t.Fatalf("SafeToClear = true, want false while the holder is alive")
	}
	if v.DroughtKind != DroughtFrozenLock {
		t.Fatalf("DroughtKind = %q, want %q", v.DroughtKind, DroughtFrozenLock)
	}
	if v.Event != EventLockHeldLive {
		t.Fatalf("Event = %q, want %q", v.Event, EventLockHeldLive)
	}
	// A recorded-but-DEAD holder must clear the veto and land the wedge.
	in.HolderAlive = false
	if v := ClassifyStaleLockWedge(in); v.State != WedgeFrozen || !v.SafeToClear {
		t.Fatalf("dead holder: state=%q safe=%v, want FROZEN_WEDGE safe=true", v.State, v.SafeToClear)
	}
}

// TestStaleLockWedgeUnconfirmed covers the resample-first guards: a single sample
// (no prior), an unreadable HEAD tail, and no active fleet must all decline to
// clear.
func TestStaleLockWedgeUnconfirmed(t *testing.T) {
	const lockMod = 1_752_437_791
	const headLog = 1_752_437_640
	base := WedgeInput{
		Prev:          LockSample{SampleUnix: lockMod + 20*60, LockPresent: true, LockModUnix: lockMod, HeadLogUnix: headLog},
		Curr:          LockSample{SampleUnix: lockMod + 26*60, LockPresent: true, LockModUnix: lockMod, HeadLogUnix: headLog},
		ActiveWorkers: 8,
		StaleAfter:    DefaultStaleLockWedgeAge,
	}

	// No prior sample: cannot prove the mtime is frozen across two samples.
	noPrev := base
	noPrev.Prev = LockSample{}
	if v := ClassifyStaleLockWedge(noPrev); v.State != WedgeUnconfirmed || v.SafeToClear {
		t.Fatalf("no-prev: state=%q safe=%v, want WEDGE_UNCONFIRMED safe=false", v.State, v.SafeToClear)
	}

	// HEAD tail unreadable in the current sample: cannot prove HEAD is frozen.
	noHead := base
	noHead.Curr.HeadLogUnix = 0
	if v := ClassifyStaleLockWedge(noHead); v.State != WedgeUnconfirmed || v.SafeToClear {
		t.Fatalf("no-head: state=%q safe=%v, want WEDGE_UNCONFIRMED safe=false", v.State, v.SafeToClear)
	}

	// No active fleet: a stale frozen lock, but no wedge to recover.
	idle := base
	idle.ActiveWorkers = 0
	if v := ClassifyStaleLockWedge(idle); v.State != WedgeUnconfirmed || v.SafeToClear {
		t.Fatalf("idle: state=%q safe=%v, want WEDGE_UNCONFIRMED safe=false", v.State, v.SafeToClear)
	}
}

// TestStaleLockWedgeClearAndFragment covers the no-lock path and pins the
// observability fragment bytes for the wedge and clear cases.
func TestStaleLockWedgeClearAndFragment(t *testing.T) {
	clear := ClassifyStaleLockWedge(WedgeInput{
		Prev:          LockSample{SampleUnix: 100, HeadLogUnix: 90},
		Curr:          LockSample{SampleUnix: 160, HeadLogUnix: 150},
		ActiveWorkers: 3,
		StaleAfter:    DefaultStaleLockWedgeAge,
	})
	if clear.State != WedgeClear || clear.SafeToClear {
		t.Fatalf("no-lock: state=%q safe=%v, want CLEAR safe=false", clear.State, clear.SafeToClear)
	}
	if got, want := StaleLockWedgeFragment(clear), "lock=CLEAR age=0s drought=none workers=3 clear=no"; got != want {
		t.Fatalf("clear fragment = %q, want %q", got, want)
	}

	const lockMod = 1_752_437_791
	const headLog = 1_752_437_640
	wedge := ClassifyStaleLockWedge(WedgeInput{
		Prev:          LockSample{SampleUnix: lockMod + 20*60, LockPresent: true, LockModUnix: lockMod, HeadLogUnix: headLog},
		Curr:          LockSample{SampleUnix: lockMod + 26*60, LockPresent: true, LockModUnix: lockMod, HeadLogUnix: headLog},
		ActiveWorkers: 8,
		StaleAfter:    DefaultStaleLockWedgeAge,
	})
	if got, want := StaleLockWedgeFragment(wedge), "lock=FROZEN_WEDGE age=1560s drought=frozen-lock-wedge workers=8 clear=yes"; got != want {
		t.Fatalf("wedge fragment = %q, want %q", got, want)
	}
}

// TestStaleLockWedgeClockSkew: a lock mtime in the future (skewed clock) floors
// the age to 0 rather than reporting a negative age, and cannot be a wedge.
func TestStaleLockWedgeClockSkew(t *testing.T) {
	const base = 1_752_400_000
	v := ClassifyStaleLockWedge(WedgeInput{
		Prev:          LockSample{SampleUnix: base, LockPresent: true, LockModUnix: base + 500, HeadLogUnix: base},
		Curr:          LockSample{SampleUnix: base + 60, LockPresent: true, LockModUnix: base + 500, HeadLogUnix: base},
		ActiveWorkers: 8,
		StaleAfter:    DefaultStaleLockWedgeAge,
	})
	if v.LockAgeSeconds != 0 {
		t.Fatalf("LockAgeSeconds = %d, want 0 for a future-dated lock", v.LockAgeSeconds)
	}
	if v.SafeToClear {
		t.Fatalf("SafeToClear = true, want false for a sub-threshold (skewed) age")
	}
	if v.State != WedgeLive {
		t.Fatalf("state = %q, want %q", v.State, WedgeLive)
	}
}
