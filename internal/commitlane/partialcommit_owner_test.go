package commitlane

import (
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// incidentNow anchors the partial-commit owner tests on the 2026-09-26 incident day.
var incidentNow = time.Date(2026, 9, 26, 12, 0, 0, 0, time.UTC)

// frozenSamples is an index.lock whose two settle samples agree: nobody is writing it.
func frozenSamples(mod time.Time) []FileFact {
	return []FileFact{
		{Exists: true, ModTime: mod, Size: 1_290_000},
		{Exists: true, ModTime: mod, Size: 1_290_000},
	}
}

// partialCommitStatus runs Status over a fake repo whose index.lock answers indexSamples
// (settle pause recorded, not served) and whose gitdir holds one next-index-<pid>.lock per
// entry of next, with the given raw mtime. alive names the pids still running.
func partialCommitStatus(t *testing.T, indexSamples []FileFact, next map[int]time.Time, alive map[int]bool, procs []Process, procErr error) Report {
	t.Helper()
	root, gitDir := testRepoPaths(t)
	idx := &settleStat{lockPath: filepath.Join(gitDir, "index.lock"), facts: indexSamples}
	nextFacts := map[string]FileFact{}
	var paths []string
	for pid, mod := range next {
		p := filepath.Join(gitDir, safecommit.NextIndexLockName(pid))
		nextFacts[p] = FileFact{Exists: true, ModTime: mod, Size: 4096}
		paths = append(paths, p)
	}
	rep, err := Status(context.Background(), Options{
		Runner:    fakeRepoRunner(root, gitDir),
		ProbeLock: func(path string) safecommit.LockProbe { return safecommit.LockProbe{Path: path} },
		Glob:      func(string) ([]string, error) { return paths, nil },
		Stat: func(path string) FileFact {
			if f, ok := nextFacts[path]; ok {
				return f
			}
			return idx.stat(path)
		},
		PIDAlive:    func(pid int) bool { return alive[pid] },
		ProcessList: func(context.Context) ([]Process, error) { return procs, procErr },
		Now:         func() time.Time { return incidentNow },
		Sleep:       func(time.Duration) {},
	})
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	return rep
}

func nextIndexDecisionFor(t *testing.T, ds []NextIndexReclaim, pid int) NextIndexReclaim {
	t.Helper()
	for _, d := range ds {
		if d.PID == pid {
			return d
		}
	}
	t.Fatalf("no next-index decision for pid %d in %+v", pid, ds)
	return NextIndexReclaim{}
}

// unrelatedGitWriter is a live `git commit` in some OTHER checkout: a by-name inventory
// match that is not, and cannot prove it is, this lock's holder.
var unrelatedGitWriter = []Process{
	{PID: 5150, Name: "git.exe", Command: `C:\Program Files\Git\cmd\git.exe commit -- C:\elsewhere\other.go`},
}

// TestStatusIncidentPartialCommitOrphanReapsDespiteUnrelatedWriter replays the 2026-09-26
// incident. A timed-out `fak commit` had its `git commit -- <paths>` (pid 111072) killed
// mid-hook, leaving .git/index.lock plus next-index-111072.lock (written just after the
// lock). Status called the lane busy ("a matching live writer was found") and the reclaim
// refused keep_live_writer because an unrelated git process matched by name — although
// the lock's named owner was dead. The lock must now read as an orphan owned by dead pid
// 111072, reap_owner_dead on the first attempt, and take its dead partner with it, while
// unrelated older residue keeps today's gates.
func TestStatusIncidentPartialCommitOrphanReapsDespiteUnrelatedWriter(t *testing.T) {
	indexMod := incidentNow.Add(-2 * time.Minute)
	rep := partialCommitStatus(t, frozenSamples(indexMod),
		map[int]time.Time{
			111072: indexMod.Add(40 * time.Millisecond), // the killed writer's temp index
			98000:  indexMod.Add(-10 * time.Minute),     // older residue from someone else
		},
		nil, unrelatedGitWriter, nil)

	lock := rep.IndexLock
	if !lock.Present || !lock.FrozenHint || lock.Advancing || lock.StaleHint {
		t.Fatalf("index lock facts = %+v, want present, frozen, not advancing, not yet stale", lock)
	}
	if lock.OwnerPID != 111072 || !lock.OwnerDead {
		t.Fatalf("owner = pid %d dead=%v, want pid 111072 dead", lock.OwnerPID, lock.OwnerDead)
	}
	if len(rep.LiveWriters) != 1 {
		t.Fatalf("live writers = %+v, want the one unrelated writer so the veto is exercised", rep.LiveWriters)
	}
	if rep.OK || rep.Verdict != VerdictStale {
		t.Fatalf("verdict = %s ok=%v (%s), want not-ok stale", rep.Verdict, rep.OK, rep.Reason)
	}
	for _, want := range []string{"orphaned", "PID 111072", "next-index-111072.lock", "dead"} {
		if !strings.Contains(rep.Reason, want) {
			t.Errorf("reason %q missing %q", rep.Reason, want)
		}
	}
	if !strings.Contains(rep.NextAction, "fak commit --reclaim-stale-index-lock --apply") {
		t.Errorf("next action = %q, want the reclaim verb", rep.NextAction)
	}

	if d := DecideIndexLockReclaim(rep); !d.Reap || d.Reason != ReclaimReapOwnerDead {
		t.Fatalf("index.lock decision = %+v, want reap_owner_dead", d)
	}
	nd := DecideNextIndexReclaim(rep)
	if d := nextIndexDecisionFor(t, nd, 111072); !d.Reap || d.Reason != ReclaimReapOwnerDead {
		t.Errorf("partner next-index decision = %+v, want reap_owner_dead", d)
	}
	if d := nextIndexDecisionFor(t, nd, 98000); d.Reap || d.Reason != ReclaimKeepLiveWriter {
		t.Errorf("unpaired residue decision = %+v, want today's keep_live_writer", d)
	}

	raw, err := json.Marshal(rep.IndexLock)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"owner_pid":111072`, `"owner_dead":true`} {
		if !strings.Contains(string(raw), want) {
			t.Errorf("index_lock JSON %s missing %s", raw, want)
		}
	}
}

// TestStatusPartialCommitOwnerNegatives pins what must NOT reap: residue that predates the
// lock (a live peer's newer lock), a live pairing writer, and a lock not frozen long enough.
func TestStatusPartialCommitOwnerNegatives(t *testing.T) {
	t.Run("residue older than index.lock names no owner", func(t *testing.T) {
		indexMod := incidentNow.Add(-2 * time.Minute)
		rep := partialCommitStatus(t, frozenSamples(indexMod),
			map[int]time.Time{111072: indexMod.Add(-1 * time.Second)}, nil, nil, nil)
		if rep.IndexLock.OwnerPID != 0 || rep.IndexLock.OwnerDead {
			t.Fatalf("owner = %d dead=%v, want none: older residue cannot own a newer lock", rep.IndexLock.OwnerPID, rep.IndexLock.OwnerDead)
		}
		if d := DecideIndexLockReclaim(rep); d.Reap || d.Reason != ReclaimKeepFresh {
			t.Fatalf("index.lock decision = %+v, want keep_fresh", d)
		}
		if d := nextIndexDecisionFor(t, DecideNextIndexReclaim(rep), 111072); d.Reap || d.Reason != ReclaimKeepFresh {
			t.Fatalf("residue decision = %+v, want keep_fresh", d)
		}
		if !rep.OK || rep.Verdict != VerdictBusy {
			t.Fatalf("verdict = %s ok=%v, want ok busy", rep.Verdict, rep.OK)
		}
	})

	t.Run("a live pairing writer is a live owner", func(t *testing.T) {
		indexMod := incidentNow.Add(-2 * time.Minute)
		rep := partialCommitStatus(t, frozenSamples(indexMod),
			map[int]time.Time{111072: indexMod.Add(40 * time.Millisecond)},
			map[int]bool{111072: true}, nil, nil)
		if rep.IndexLock.OwnerPID != 111072 || rep.IndexLock.OwnerDead {
			t.Fatalf("owner = %d dead=%v, want live pid 111072", rep.IndexLock.OwnerPID, rep.IndexLock.OwnerDead)
		}
		if d := DecideIndexLockReclaim(rep); d.Reap || d.Reason != ReclaimKeepLiveOwner {
			t.Fatalf("index.lock decision = %+v, want keep_live_owner", d)
		}
		if d := nextIndexDecisionFor(t, DecideNextIndexReclaim(rep), 111072); d.Reap || d.Reason != ReclaimKeepLiveOwner {
			t.Fatalf("partner decision = %+v, want keep_live_owner", d)
		}
		if rep.Verdict != VerdictBusy || !strings.Contains(rep.Reason, "held by live partial commit PID 111072") {
			t.Fatalf("verdict = %s (%s), want busy held by live partial commit PID 111072", rep.Verdict, rep.Reason)
		}
	})

	t.Run("any live pairing writer outranks a dead earlier one", func(t *testing.T) {
		indexMod := incidentNow.Add(-2 * time.Minute)
		rep := partialCommitStatus(t, frozenSamples(indexMod),
			map[int]time.Time{
				111072: indexMod.Add(40 * time.Millisecond),
				111500: indexMod.Add(80 * time.Millisecond),
			},
			map[int]bool{111500: true}, nil, nil)
		if rep.IndexLock.OwnerDead {
			t.Fatalf("owner dead with a live pairing writer: %+v", rep.IndexLock)
		}
		if d := DecideIndexLockReclaim(rep); d.Reap {
			t.Fatalf("index.lock decision = %+v, want keep", d)
		}
		if d := nextIndexDecisionFor(t, DecideNextIndexReclaim(rep), 111072); d.Reap {
			t.Fatalf("dead pairing file reaped while a live pairing writer exists: %+v", d)
		}
	})

	t.Run("frozen less than the owner-dead window does not reap", func(t *testing.T) {
		indexMod := incidentNow.Add(-30 * time.Second)
		rep := partialCommitStatus(t, frozenSamples(indexMod),
			map[int]time.Time{111072: indexMod.Add(40 * time.Millisecond)}, nil, nil, nil)
		if rep.IndexLock.OwnerPID != 111072 || !rep.IndexLock.OwnerDead || rep.IndexLock.FrozenHint {
			t.Fatalf("index lock = %+v, want dead owner 111072 but not yet frozen", rep.IndexLock)
		}
		if d := DecideIndexLockReclaim(rep); d.Reap {
			t.Fatalf("index.lock decision = %+v, want keep inside %s", d, DefaultOwnerDeadIndexAge)
		}
		if d := nextIndexDecisionFor(t, DecideNextIndexReclaim(rep), 111072); d.Reap || d.Reason != ReclaimKeepFresh {
			t.Fatalf("partner decision = %+v, want keep_fresh", d)
		}
		if rep.Verdict != VerdictBusy || !strings.Contains(rep.Reason, "not yet frozen") {
			t.Fatalf("verdict = %s (%s), want busy naming the not-yet-frozen dead owner", rep.Verdict, rep.Reason)
		}
	})

	t.Run("an advancing lock is never an orphan", func(t *testing.T) {
		indexMod := incidentNow.Add(-2 * time.Minute)
		rep := partialCommitStatus(t,
			[]FileFact{
				{Exists: true, ModTime: indexMod, Size: 100},
				{Exists: true, ModTime: indexMod.Add(time.Second), Size: 4096},
			},
			map[int]time.Time{111072: indexMod.Add(2 * time.Second)}, nil, nil, nil)
		if !rep.IndexLock.Advancing {
			t.Fatalf("index lock = %+v, want advancing", rep.IndexLock)
		}
		if d := DecideIndexLockReclaim(rep); d.Reap || d.Reason != ReclaimKeepAdvancing {
			t.Fatalf("index.lock decision = %+v, want keep_advancing", d)
		}
		if d := nextIndexDecisionFor(t, DecideNextIndexReclaim(rep), 111072); d.Reap {
			t.Fatalf("partner of an advancing lock reaped: %+v", d)
		}
		if rep.Verdict != VerdictBusy {
			t.Fatalf("verdict = %s (%s), want busy", rep.Verdict, rep.Reason)
		}
	})
}

// TestDecidePartialCommitOwnerDeadIsConjunctive pins the pairing proof at the decision
// layer: it reaps the lock and its named partner through a failed probe and an unrelated
// writer, and flipping any one half — freeze, dead owner, named pid, presence, or a moving
// mtime — keeps both. The partner's own live pid still vetoes, and other rows keep today's
// gates.
func TestDecidePartialCommitOwnerDeadIsConjunctive(t *testing.T) {
	base := func() Report {
		return Report{
			ProcessProbe: "error",
			LiveWriters:  []ProcessFact{{PID: 5150, Match: "git_writer"}},
			IndexLock: IndexLock{
				Path: "/g/index.lock", Present: true, FrozenHint: true,
				OwnerPID: 111072, OwnerDead: true,
			},
			NextIndexLocks: []NextIndexLock{
				{Path: "/g/next-index-111072.lock", PID: 111072},
				{Path: "/g/next-index-98000.lock", PID: 98000, StaleHint: true},
			},
		}
	}
	rep := base()
	if d := DecideIndexLockReclaim(rep); !d.Reap || d.Reason != ReclaimReapOwnerDead {
		t.Fatalf("baseline index.lock = %+v, want reap_owner_dead", d)
	}
	nd := DecideNextIndexReclaim(rep)
	if !nd[0].Reap || nd[0].Reason != ReclaimReapOwnerDead {
		t.Fatalf("baseline partner = %+v, want reap_owner_dead", nd[0])
	}
	if nd[1].Reap || nd[1].Reason != ReclaimKeepProbeFailed {
		t.Fatalf("unpaired row = %+v, want today's keep_probe_failed", nd[1])
	}

	keeps := map[string]func(*Report){
		"not frozen":     func(r *Report) { r.IndexLock.FrozenHint = false },
		"owner alive":    func(r *Report) { r.IndexLock.OwnerDead = false },
		"no named owner": func(r *Report) { r.IndexLock.OwnerPID = 0 },
		"advancing":      func(r *Report) { r.IndexLock.Advancing = true },
		"lock absent":    func(r *Report) { r.IndexLock.Present = false },
	}
	for name, mut := range keeps {
		r := base()
		mut(&r)
		if d := DecideIndexLockReclaim(r); d.Reap {
			t.Errorf("%s: index.lock reaped: %+v", name, d)
		}
		if d := DecideNextIndexReclaim(r)[0]; d.Reap {
			t.Errorf("%s: partner reaped: %+v", name, d)
		}
	}

	r := base()
	r.NextIndexLocks[0].OwnerAlive = true
	if d := DecideNextIndexReclaim(r)[0]; d.Reap {
		t.Errorf("a live partner pid must veto its own reap: %+v", d)
	}

	adv := base()
	adv.IndexLock.Advancing = true
	finalize(&adv)
	if adv.Verdict == VerdictStale {
		t.Errorf("an advancing lock must not be reported as an orphan: %s (%s)", adv.Verdict, adv.Reason)
	}
}

// TestStatusBusyReasonClaimsALiveWriterOnlyWhenOneWasFound fixes the misleading busy text:
// a present index.lock with an empty inventory must say no writer was found, and only a
// listed writer earns "a matching live writer was found".
func TestStatusBusyReasonClaimsALiveWriterOnlyWhenOneWasFound(t *testing.T) {
	indexMod := incidentNow.Add(-30 * time.Second)

	rep := partialCommitStatus(t, frozenSamples(indexMod), nil, nil, nil, nil)
	if rep.Verdict != VerdictBusy {
		t.Fatalf("verdict = %s, want busy", rep.Verdict)
	}
	if strings.Contains(rep.Reason, "a matching live writer was found") {
		t.Fatalf("reason %q claims a live writer with an empty inventory", rep.Reason)
	}
	for _, want := range []string{"age 30s", "no matching live writer found", "younger than the stale window"} {
		if !strings.Contains(rep.Reason, want) {
			t.Errorf("reason %q missing %q", rep.Reason, want)
		}
	}

	rep = partialCommitStatus(t, frozenSamples(indexMod), nil, nil, nil, errProcessProbe)
	if !strings.Contains(rep.Reason, "live writers are unknown") {
		t.Errorf("failed inventory reason = %q, want it to say writers are unknown", rep.Reason)
	}

	rep = partialCommitStatus(t, frozenSamples(indexMod), nil, nil, unrelatedGitWriter, nil)
	if !strings.Contains(rep.Reason, "a matching live writer was found") {
		t.Errorf("listed writer reason = %q, want the live-writer wording", rep.Reason)
	}
}
