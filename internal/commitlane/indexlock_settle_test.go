package commitlane

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// settleStat is a Stat seam that answers a scripted SEQUENCE of facts for .git/index.lock
// while every other path (the sibling fak-commit.lock, the next-index glob) reads absent.
// The sequence is what lets a test drive the observer's two-sample settle witness: entry 0
// is the first sample, entry 1 the sample taken after the settle window. It also counts the
// samples, so a test can prove the second one actually happened rather than assuming it.
type settleStat struct {
	lockPath string
	facts    []FileFact
	calls    int
}

func (s *settleStat) stat(path string) FileFact {
	if path != s.lockPath {
		return FileFact{}
	}
	i := s.calls
	s.calls++
	if i >= len(s.facts) {
		i = len(s.facts) - 1
	}
	return s.facts[i]
}

// settleSleeper records the settle pauses the observer asks for WITHOUT taking them, so
// every test in this file exercises the real two-sample path at zero wall-clock cost. A
// witness that had to sleep to be proven could not be bounded, which is precisely the
// property #5335 is about.
type settleSleeper struct{ waits []time.Duration }

func (s *settleSleeper) sleep(d time.Duration) { s.waits = append(s.waits, d) }

// settleStatus runs Status over a fake repo whose index.lock answers the given sample
// sequence, with the settle pause recorded rather than served.
func settleStatus(t *testing.T, now time.Time, facts []FileFact, commitLock safecommit.LockProbe, procs []Process) (Report, *settleSleeper) {
	t.Helper()
	root, gitDir := testRepoPaths(t)
	stat := &settleStat{lockPath: filepath.Join(gitDir, "index.lock"), facts: facts}
	sleeper := &settleSleeper{}
	rep, err := Status(context.Background(), Options{
		Runner: fakeRepoRunner(root, gitDir),
		ProbeLock: func(path string) safecommit.LockProbe {
			p := commitLock
			p.Path = path
			return p
		},
		Stat:        stat.stat,
		ProcessList: func(context.Context) ([]Process, error) { return procs, nil },
		Now:         func() time.Time { return now },
		Sleep:       sleeper.sleep,
	})
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if stat.calls < 2 {
		t.Fatalf("index.lock sampled %d time(s); the settle witness needs two samples", stat.calls)
	}
	return rep, sleeper
}

// TestStatusWitnessesAdvancingIndexLock is the guard the age-only reap was missing: a lock
// whose mtime is ADVANCING is being written right now, so it must never be reaped even
// though its first sample reads far past the stale grace window. Before the settle witness
// existed, this exact shape — a slow live writer holding index.lock with an old-looking
// mtime — reaped out from under the writer.
func TestStatusWitnessesAdvancingIndexLock(t *testing.T) {
	now := time.Date(2026, 7, 28, 12, 0, 0, 0, time.UTC)
	old := now.Add(-40 * time.Minute)
	rep, sleeper := settleStatus(t, now,
		[]FileFact{
			{Exists: true, ModTime: old, Size: 1_290_000},
			{Exists: true, ModTime: old.Add(2 * time.Second), Size: 1_320_000},
		},
		safecommit.LockProbe{}, nil)

	if !rep.IndexLock.Advancing {
		t.Fatalf("a lock whose mtime moved across the settle window must read Advancing: %+v", rep.IndexLock)
	}
	if len(sleeper.waits) != 1 || sleeper.waits[0] != DefaultIndexLockSettleWindow {
		t.Fatalf("settle waits = %v, want exactly one %s pause", sleeper.waits, DefaultIndexLockSettleWindow)
	}
	if rep.IndexLock.SettleMillis != int64(DefaultIndexLockSettleWindow/time.Millisecond) {
		t.Fatalf("SettleMillis = %d, want the window actually spent", rep.IndexLock.SettleMillis)
	}
	if rep.IndexLock.FrozenHint {
		t.Fatalf("an advancing lock is not frozen: %+v", rep.IndexLock)
	}
	if d := DecideIndexLockReclaim(rep); d.Reap || d.Reason != ReclaimKeepAdvancing {
		t.Fatalf("advancing lock decision = %+v, want keep_advancing", d)
	}
}

// TestStatusWitnessesAdvancingIndexLockBySizeAlone covers the writer that grows the lock
// within one filesystem mtime tick: the size moved, so the lock is still being written.
func TestStatusWitnessesAdvancingIndexLockBySizeAlone(t *testing.T) {
	now := time.Date(2026, 7, 28, 12, 0, 0, 0, time.UTC)
	frozen := now.Add(-40 * time.Minute)
	rep, _ := settleStatus(t, now,
		[]FileFact{
			{Exists: true, ModTime: frozen, Size: 100},
			{Exists: true, ModTime: frozen, Size: 4096},
		},
		safecommit.LockProbe{}, nil)

	if !rep.IndexLock.Advancing {
		t.Fatalf("a growing lock must read Advancing: %+v", rep.IndexLock)
	}
	if d := DecideIndexLockReclaim(rep); d.Reap {
		t.Fatalf("a growing lock must not be reaped: %+v", d)
	}
}

// TestStatusFrozenIndexLockWithDeadFakOwnerReapsOnTheFirstAttempt is #5335 item 3's
// definition of done. A `fak commit` killed at its tool timeout leaves .git/index.lock plus
// a sibling fak-commit.lock naming its own now-dead pid. The lock is frozen for a minute but
// nowhere near the fifteen-minute grace window, and a peer `git add -A` swarm keeps a
// by-name live writer on the inventory at essentially all times. That combination used to
// refuse — the reclaim only succeeded when a retry happened to sample a gap in the swarm —
// so the lane stayed wedged while the swarm manufactured the next orphan.
func TestStatusFrozenIndexLockWithDeadFakOwnerReapsOnTheFirstAttempt(t *testing.T) {
	now := time.Date(2026, 7, 28, 12, 0, 0, 0, time.UTC)
	frozen := now.Add(-90 * time.Second)
	root, _ := testRepoPaths(t)
	swarm := []Process{
		{PID: 7304, Name: "git.exe", Command: `C:\Program Files\Git\cmd\git.exe add -A ` + root},
		{PID: 75256, Name: "git.exe", Command: `C:\Program Files\Git\cmd\git.exe add -A ` + root},
	}
	rep, _ := settleStatus(t, now,
		[]FileFact{
			{Exists: true, ModTime: frozen, Size: 1_290_000},
			{Exists: true, ModTime: frozen, Size: 1_290_000},
		},
		safecommit.LockProbe{Exists: true, HolderPID: 21688, Alive: false, Stale: true},
		swarm)

	if !rep.IndexLock.FrozenHint || rep.IndexLock.Advancing {
		t.Fatalf("index lock facts = %+v, want frozen and not advancing", rep.IndexLock)
	}
	if rep.IndexLock.StaleHint {
		t.Fatalf("90s is inside the %s grace window; the reap must not lean on staleness: %+v", DefaultStaleIndexAge, rep.IndexLock)
	}
	if len(rep.LiveWriters) == 0 {
		t.Fatalf("the swarm must be visible as live writers, else the test does not exercise the veto")
	}
	d := DecideIndexLockReclaim(rep)
	if !d.Reap || d.Reason != ReclaimReapOwnerDead {
		t.Fatalf("decision = %+v, want a first-attempt reap_owner_dead", d)
	}
}

// TestDecideIndexLockReclaimOwnerDeadEvidenceIsConjunctive pins that the owner-dead reap
// needs BOTH halves — the freeze and the dead named creator — and that a lock actively
// being written is kept even when both halves are otherwise satisfied.
func TestDecideIndexLockReclaimOwnerDeadEvidenceIsConjunctive(t *testing.T) {
	base := Report{
		ProcessProbe: "ok",
		CommitLock:   CommitLock{Path: "/g/fak-commit.lock", Present: true, HolderPID: 21688, Stale: true},
		IndexLock:    IndexLock{Path: "/g/index.lock", Present: true, FrozenHint: true},
		LiveWriters:  []ProcessFact{{PID: 4242, Match: "git_writer"}},
	}
	if d := DecideIndexLockReclaim(base); !d.Reap || d.Reason != ReclaimReapOwnerDead {
		t.Fatalf("baseline = %+v, want reap_owner_dead", d)
	}

	keeps := map[string]func(*Report){
		"not frozen yet":            func(r *Report) { r.IndexLock.FrozenHint = false },
		"sibling owner still alive": func(r *Report) { r.CommitLock.Stale = false; r.CommitLock.HolderAlive = true },
		"sibling names no pid":      func(r *Report) { r.CommitLock.HolderPID = 0; r.CommitLock.Stale = false },
		"no sibling lock at all":    func(r *Report) { r.CommitLock = CommitLock{} },
		"lock is advancing":         func(r *Report) { r.IndexLock.Advancing = true },
	}
	for name, mut := range keeps {
		r := base
		mut(&r)
		if d := DecideIndexLockReclaim(r); d.Reap {
			t.Errorf("%s: expected keep, but the decision reaped (%+v)", name, d)
		}
	}
}

// TestDecideIndexLockReclaimKeepsAdvancingLockPastEveryAgeGate proves the advancing witness
// outranks BOTH age gates. Age is an inference about a holder; a moving mtime is an
// observation of one, so the observation must win — otherwise the shorter owner-dead window
// introduced here would widen the blast radius of a misread instead of narrowing it.
func TestDecideIndexLockReclaimKeepsAdvancingLockPastEveryAgeGate(t *testing.T) {
	advancing := Report{
		ProcessProbe: "ok",
		CommitLock:   CommitLock{Present: true, HolderPID: 21688, Stale: true},
		IndexLock: IndexLock{
			Path:       "/g/index.lock",
			Present:    true,
			StaleHint:  true,
			FrozenHint: true,
			Advancing:  true,
		},
	}
	d := DecideIndexLockReclaim(advancing)
	if d.Reap || d.Reason != ReclaimKeepAdvancing {
		t.Fatalf("decision = %+v, want keep_advancing even with both age gates satisfied", d)
	}
}

// TestStatusDropsIndexLockThatClearedDuringSettle: the lock released between the two
// samples. Reporting it as present would hand the actuator a path whose next occupant is
// whatever live writer grabs the lane next.
func TestStatusDropsIndexLockThatClearedDuringSettle(t *testing.T) {
	now := time.Date(2026, 7, 28, 12, 0, 0, 0, time.UTC)
	rep, _ := settleStatus(t, now,
		[]FileFact{
			{Exists: true, ModTime: now.Add(-40 * time.Minute), Size: 1_290_000},
			{},
		},
		safecommit.LockProbe{Exists: true, HolderPID: 21688, Alive: false, Stale: true}, nil)

	if rep.IndexLock.Present {
		t.Fatalf("a lock that vanished inside the settle window must not read present: %+v", rep.IndexLock)
	}
	if d := DecideIndexLockReclaim(rep); d.Reap || d.Reason != ReclaimKeepAbsent {
		t.Fatalf("decision = %+v, want keep_absent", d)
	}
}

// TestReclaimConvergesOnTheFirstPassAgainstATransientWriterSwarm replays the measured
// failure: fifteen consecutive reclaim attempts against a frozen orphan while transient
// `git add -A` spawns come and go, only one of which samples a gap in the swarm. The
// decision must reap on pass 1 rather than waiting for the swarm to blink. The loop is a
// fixed fifteen iterations with no waiting of any kind, so it can fail but never hang.
func TestReclaimConvergesOnTheFirstPassAgainstATransientWriterSwarm(t *testing.T) {
	const passes = 15
	firstReap := -1
	for i := 0; i < passes; i++ {
		rep := Report{
			ProcessProbe: "ok",
			CommitLock:   CommitLock{Present: true, HolderPID: 21688, Stale: true},
			IndexLock:    IndexLock{Path: "/g/index.lock", Present: true, FrozenHint: true},
		}
		// The swarm is present on every pass but the last — the "15th rapid retry" gap.
		if i < passes-1 {
			rep.LiveWriters = []ProcessFact{{PID: 7304 + i, Match: "git_writer"}}
		}
		if DecideIndexLockReclaim(rep).Reap && firstReap < 0 {
			firstReap = i
		}
	}
	switch {
	case firstReap < 0:
		t.Fatalf("no reap in %d passes: the reclaim never clears a frozen orphan whose named creator is dead", passes)
	case firstReap != 0:
		t.Fatalf("first reap on pass %d, want pass 1: the reclaim still waits for a gap in the swarm", firstReap+1)
	}
}

// TestReclaimNeverReapsAGenuinelyHeldLockAcrossTheSwarm is the other half of the same
// contract: the same fifteen passes against a lock a live holder keeps touching must reap
// on NONE of them, whatever the inventory shows.
func TestReclaimNeverReapsAGenuinelyHeldLockAcrossTheSwarm(t *testing.T) {
	for i := 0; i < 15; i++ {
		rep := Report{
			ProcessProbe: "ok",
			CommitLock:   CommitLock{Present: true, HolderPID: 21688, Stale: true},
			IndexLock: IndexLock{
				Path:       "/g/index.lock",
				Present:    true,
				FrozenHint: true,
				Advancing:  true,
			},
		}
		if i%2 == 0 {
			rep.LiveWriters = []ProcessFact{{PID: 7304 + i, Match: "git_writer"}}
		}
		if d := DecideIndexLockReclaim(rep); d.Reap {
			t.Fatalf("pass %d reaped a lock whose mtime is still advancing: %+v", i+1, d)
		}
	}
}

// TestStatusSettleWindowIsSpentOnlyWhenALockIsPresent keeps the witness off the hot path:
// the clear lane — every commit that is not wedged — must not pay the settle pause.
func TestStatusSettleWindowIsSpentOnlyWhenALockIsPresent(t *testing.T) {
	root, gitDir := testRepoPaths(t)
	sleeper := &settleSleeper{}
	stat := &settleStat{lockPath: filepath.Join(gitDir, "index.lock"), facts: []FileFact{{}}}
	rep, err := Status(context.Background(), Options{
		Runner:      fakeRepoRunner(root, gitDir),
		ProbeLock:   func(path string) safecommit.LockProbe { return safecommit.LockProbe{Path: path} },
		Stat:        stat.stat,
		ProcessList: func(context.Context) ([]Process, error) { return nil, nil },
		Now:         fixedNow,
		Sleep:       sleeper.sleep,
	})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Verdict != VerdictClear {
		t.Fatalf("verdict = %q, want clear", rep.Verdict)
	}
	if len(sleeper.waits) != 0 {
		t.Fatalf("the clear lane spent %v settling; it must spend nothing", sleeper.waits)
	}
}
