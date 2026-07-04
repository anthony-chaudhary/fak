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

// TestRenewExtendsOwnLockKeepingItLiveAcrossTTL proves the heartbeat: a holder
// that renews before its TTL keeps the lock live, so a concurrent cutter still
// sees it held and cannot steal-stale it mid-ship — the collision #1391 stops.
func TestRenewExtendsOwnLockKeepingItLiveAcrossTTL(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_RELEASE_LOCK_ROOT", dir)
	t0 := time.Unix(5_000_000, 0)

	ship := optsAt(dir, "ship-session", t0)
	if _, err := Acquire(ship, 30*time.Minute, "cut", false, false); err != nil {
		t.Fatalf("acquire failed: %v", err)
	}

	// 20 min in — before the 30-min TTL — the ship heartbeat renews for another
	// 30 min. acquired_at must be preserved; expiry must move to t1+30m.
	t1 := t0.Add(20 * time.Minute)
	st, err := Renew(optsAt(dir, "ship-session", t1), 30*time.Minute, false)
	if err != nil {
		t.Fatalf("renew by owner failed: %v", err)
	}
	if st == nil || !st.Held || st.Lock == nil || st.Lock.AcquiredAt != epoch(t0) {
		t.Fatalf("renew should preserve acquired_at and stay held; got %+v", st)
	}
	if want := epoch(t1) + (30 * time.Minute).Seconds(); st.Lock.ExpiresAt != want {
		t.Fatalf("renew expiry = %v, want %v", st.Lock.ExpiresAt, want)
	}

	// 35 min after the ORIGINAL acquire — which without the renew would be stale —
	// a concurrent cutter must still be refused, because the renew pushed expiry
	// out to t1+30m = t0+50m.
	t2 := t0.Add(35 * time.Minute)
	if s := Status(optsAt(dir, "probe", t2)); s.Stale {
		t.Fatalf("renewed lock must not be stale at t0+35m; got %+v", s)
	}
	bot := optsAt(dir, "cadence-bot", t2)
	if _, err := Acquire(bot, 30*time.Minute, "auto", true /*steal stale*/, false); !errors.Is(err, ErrHeld) {
		t.Fatalf("a renewed (live) lock must refuse a steal-stale acquire; got %v", err)
	}
}

// TestRenewRefusesWhenLockStolen proves the theft signal: if a peer already owns
// the lock (a missed renewal let it go stale and get stolen), Renew refuses with
// ErrRenewLost instead of silently clobbering the peer's lease.
func TestRenewRefusesWhenLockStolen(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_RELEASE_LOCK_ROOT", dir)
	now := time.Unix(6_000_000, 0)

	peer := optsAt(dir, "peer-session", now)
	if _, err := Acquire(peer, 30*time.Minute, "peer cut", false, false); err != nil {
		t.Fatalf("peer acquire failed: %v", err)
	}

	// The original holder tries to heartbeat a lock a peer now owns.
	st, err := Renew(optsAt(dir, "original-session", now), 30*time.Minute, false)
	if !errors.Is(err, ErrRenewLost) {
		t.Fatalf("renew over a peer's lock: want ErrRenewLost, got err=%v state=%+v", err, st)
	}
	if st == nil || st.Lock == nil || st.Lock.Owner != "peer-session" {
		t.Fatalf("refusal should report the live holder; got %+v", st)
	}
	// The peer's lock is untouched.
	if s := Status(peer); !s.Held || s.Lock.Owner != "peer-session" {
		t.Fatalf("peer lock must survive a refused renew; got %+v", s)
	}
}

// TestRenewGoneRefusedUnlessForced proves renewing an absent lock refuses by
// default (the lease is gone — do not resurrect it under the caller), but a
// forced renew re-establishes it.
func TestRenewGoneRefusedUnlessForced(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_RELEASE_LOCK_ROOT", dir)
	now := time.Unix(7_000_000, 0)

	o := optsAt(dir, "session", now)
	if _, err := Renew(o, 30*time.Minute, false); !errors.Is(err, ErrRenewLost) {
		t.Fatalf("renew of absent lock: want ErrRenewLost, got %v", err)
	}
	st, err := Renew(o, 30*time.Minute, true)
	if err != nil {
		t.Fatalf("forced renew of absent lock should establish it: %v", err)
	}
	if st == nil || !st.Held || st.Lock == nil || st.Lock.Owner != "session" {
		t.Fatalf("forced renew should hold the lock for the caller; got %+v", st)
	}
	if s := Status(o); !s.Held || s.Lock.Owner != "session" {
		t.Fatalf("forced renew should persist; got %+v", s)
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
