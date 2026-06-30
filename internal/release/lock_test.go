package release

import (
	"errors"
	"path/filepath"
	"testing"
	"time"
)

// optsAt returns Options rooted at dir, owned by owner, with a fixed clock at t.
func optsAt(dir, owner string, t time.Time) Options {
	return Options{Root: dir, Owner: owner, Now: func() time.Time { return t }}
}

// TestTwoAcquirersMutuallyExclusive proves a live lock held by one owner refuses
// a second, different owner — the core race #1391 exists to stop (scheduled cut
// vs human /release).
func TestTwoAcquirersMutuallyExclusive(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_RELEASE_LOCK_ROOT", dir) // both contend on the same well-known file
	now := time.Unix(1_000_000, 0)

	human := optsAt(dir, "human-session", now)
	bot := optsAt(dir, "cadence-bot", now)

	if _, err := Acquire(human, 30*time.Minute, "hand cut", false, false); err != nil {
		t.Fatalf("first acquire (human) failed: %v", err)
	}

	// The unattended cadence tick fires while the human holds the lock: it must
	// be refused, not allowed to race VERSION/tag.
	st, err := Acquire(bot, 30*time.Minute, "auto cut", true /*steal stale*/, false)
	if !errors.Is(err, ErrHeld) {
		t.Fatalf("second acquire by other owner: want ErrHeld, got err=%v state=%+v", err, st)
	}
	if st == nil || st.Lock == nil || st.Lock.Owner != "human-session" {
		t.Fatalf("refusal should report the live holder; got %+v", st)
	}

	// HeldByOther is the predicate the auto-cut consults to decide to defer.
	other, holder := HeldByOther(bot)
	if !other || holder == nil || holder.Owner != "human-session" {
		t.Fatalf("HeldByOther(bot): want true held by human-session, got %v %+v", other, holder)
	}
	// The human's own session does NOT see itself as "other".
	if other, _ := HeldByOther(human); other {
		t.Fatalf("HeldByOther(human): a session must not see its own lock as foreign")
	}
}

// TestStaleLockTakenOver proves a dead holder's lock (past TTL) is stolen by a
// later acquirer with stealStale — a crashed cutter auto-recovers.
func TestStaleLockTakenOver(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_RELEASE_LOCK_ROOT", dir)
	t0 := time.Unix(2_000_000, 0)

	dead := optsAt(dir, "crashed-session", t0)
	if _, err := Acquire(dead, 30*time.Minute, "", false, false); err != nil {
		t.Fatalf("seed lock failed: %v", err)
	}

	// 31 minutes later the lock is past its 30-min TTL.
	t1 := t0.Add(31 * time.Minute)
	if st := Status(optsAt(dir, "x", t1)); !st.Stale {
		t.Fatalf("lock should be stale after TTL; got %+v", st)
	}

	bot := optsAt(dir, "cadence-bot", t1)
	st, err := Acquire(bot, 30*time.Minute, "", true /*steal stale*/, false)
	if err != nil {
		t.Fatalf("stale takeover failed: %v", err)
	}
	if st.Stolen == nil || st.Stolen.Owner != "crashed-session" {
		t.Fatalf("takeover should report the stolen lock; got %+v", st)
	}
	if st.Lock == nil || st.Lock.Owner != "cadence-bot" {
		t.Fatalf("lock should now be owned by cadence-bot; got %+v", st)
	}

	// A NON-stale lock is NOT stolen even with stealStale: confirm the steal is
	// gated on staleness, not just the flag.
	fresh := optsAt(dir, "fresh-holder", t1)
	if _, err := Release(bot, true); err != nil {
		t.Fatalf("release for non-stale setup failed: %v", err)
	}
	if _, err := Acquire(fresh, 30*time.Minute, "", false, false); err != nil {
		t.Fatalf("seed fresh lock failed: %v", err)
	}
	intruder := optsAt(dir, "intruder", t1.Add(time.Minute)) // still well within TTL
	if _, err := Acquire(intruder, 30*time.Minute, "", true, false); !errors.Is(err, ErrHeld) {
		t.Fatalf("a live lock must not be stolen by stealStale; want ErrHeld, got %v", err)
	}
}

// TestReleaseFreesLock proves Release frees the lock for the owner, and that a
// non-owner cannot release it without force.
func TestReleaseFreesLock(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_RELEASE_LOCK_ROOT", dir)
	now := time.Unix(3_000_000, 0)

	a := optsAt(dir, "session-a", now)
	b := optsAt(dir, "session-b", now)

	if _, err := Acquire(a, 30*time.Minute, "", false, false); err != nil {
		t.Fatalf("acquire failed: %v", err)
	}

	// A different session cannot release A's lock.
	if _, err := Release(b, false); !errors.Is(err, ErrNotOwner) {
		t.Fatalf("non-owner release: want ErrNotOwner, got %v", err)
	}

	// The owner frees it.
	if st, err := Release(a, false); err != nil || st.Held {
		t.Fatalf("owner release: err=%v state=%+v", err, st)
	}

	// Now the previously-blocked session can acquire cleanly.
	if _, err := Acquire(b, 30*time.Minute, "", false, false); err != nil {
		t.Fatalf("acquire after release failed: %v", err)
	}
	if st := Status(b); !st.Held || st.Lock.Owner != "session-b" {
		t.Fatalf("session-b should now hold the lock; got %+v", st)
	}

	// Releasing a free lock is a no-op success.
	if _, err := Release(b, false); err != nil {
		t.Fatalf("release b failed: %v", err)
	}
	if st, err := Release(a, false); err != nil || st.Held {
		t.Fatalf("release of absent lock should be a no-op success; err=%v st=%+v", err, st)
	}
}

// TestLockPathHonorsEnvOverride confirms the lockfile lands at the well-known
// name under the env-overridden root, so the Go and Python paths contend on the
// SAME file.
func TestLockPathHonorsEnvOverride(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_RELEASE_LOCK_ROOT", dir)
	o := optsAt("/some/other/root", "o", time.Unix(4_000_000, 0))
	if got, want := o.lockPath(), filepath.Join(dir, LockName); got != want {
		t.Fatalf("lockPath under env override = %q, want %q", got, want)
	}
}
