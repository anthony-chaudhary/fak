package safecommit

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// writeLockAt creates path with the given mtime.
func writeLockAt(t *testing.T, path string, mod time.Time) {
	t.Helper()
	writeSafecommitFile(t, path, "")
	if err := os.Chtimes(path, mod, mod); err != nil {
		t.Fatal(err)
	}
}

func stubKilledCommitAlive(t *testing.T, alive map[int]bool) {
	t.Helper()
	prev := killedCommitAlive
	killedCommitAlive = func(pid int) bool { return alive[pid] }
	t.Cleanup(func() { killedCommitAlive = prev })
}

func lockFileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// TestPartialCommitOwner pins the pairing rule both reapers share: git writes index.lock
// before it creates next-index-<pid>.lock, so only residue NOT older than the lock can name
// its creator; a live pairing writer always wins; pid-less names never pair.
func TestPartialCommitOwner(t *testing.T) {
	lock := time.Date(2026, 9, 26, 16, 18, 0, 0, time.UTC)
	cases := []struct {
		name     string
		cands    []NextIndexCandidate
		wantPID  int
		wantDead bool
		wantOK   bool
	}{
		{"incident: dead partner newer than the lock", []NextIndexCandidate{{PID: 111072, ModTime: lock.Add(time.Second)}}, 111072, true, true},
		{"same instant pairs", []NextIndexCandidate{{PID: 7, ModTime: lock}}, 7, true, true},
		{"older residue never pairs (a live peer's newer lock)", []NextIndexCandidate{{PID: 9, ModTime: lock.Add(-time.Minute)}}, 0, false, false},
		{"live pairing writer wins", []NextIndexCandidate{{PID: 5, ModTime: lock.Add(time.Second)}, {PID: 6, ModTime: lock.Add(2 * time.Second), Alive: true}}, 5, false, true},
		{"earliest pairing file names the owner", []NextIndexCandidate{{PID: 20, ModTime: lock.Add(3 * time.Second)}, {PID: 10, ModTime: lock.Add(time.Second)}}, 10, true, true},
		{"pid-less name never pairs", []NextIndexCandidate{{PID: 0, ModTime: lock.Add(time.Second)}}, 0, false, false},
		{"no residue", nil, 0, false, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pid, dead, ok := PartialCommitOwner(lock, tc.cands)
			if pid != tc.wantPID || dead != tc.wantDead || ok != tc.wantOK {
				t.Fatalf("PartialCommitOwner = (%d, %v, %v), want (%d, %v, %v)", pid, dead, ok, tc.wantPID, tc.wantDead, tc.wantOK)
			}
		})
	}
	if got := NextIndexLockPID("next-index-111072.lock"); got != 111072 {
		t.Fatalf("NextIndexLockPID = %d, want 111072", got)
	}
	if got := NextIndexLockPID("next-index-111072.lock.lock"); got != 0 {
		t.Fatalf("a hook's nested lock must not name a pid, got %d", got)
	}
}

// TestIndexRedirected: git's own answer to "where is the index" decides whether
// <gitdir>/index.lock can be ours. A redirected index (an inherited GIT_INDEX_FILE) is
// reported as such, git's forward slashes are not mistaken for a redirect, and a git that
// cannot answer is reported as unknown so each caller picks its own fail direction.
func TestIndexRedirected(t *testing.T) {
	gitDir := filepath.Join(t.TempDir(), ".git")
	answer := func(out string, code int, err error) Runner {
		return func(context.Context, string, ...string) (string, int, error) { return out, code, err }
	}
	cases := []struct {
		name                string
		run                 Runner
		wantRedir, wantKnow bool
	}{
		{"default index, git's slashes", answer(filepath.ToSlash(filepath.Join(gitDir, "index"))+"\n", 0, nil), false, true},
		{"inherited GIT_INDEX_FILE", answer(filepath.ToSlash(filepath.Join(gitDir, "index.lock"))+"\n", 0, nil), true, true},
		{"git too old for --path-format", answer("", 129, nil), false, false},
		{"git not executable", answer("", -1, os.ErrNotExist), false, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			redir, known := indexRedirected(context.Background(), tc.run, "", gitDir)
			if redir != tc.wantRedir || known != tc.wantKnow {
				t.Fatalf("indexRedirected = (%v, %v), want (%v, %v)", redir, known, tc.wantRedir, tc.wantKnow)
			}
		})
	}
}

// TestRollbackKilledCommit: after the runner kills a `git commit -- <paths>` mid-hook, it
// removes exactly the locks that commit's run left — the dead next-index-<pid>.lock created
// during the run and the index.lock that pairs with it — and nothing it cannot prove. The
// pid is deliberately unrelated to anything safecommit started: on Windows the file is named
// by the Git-for-Windows launcher's child git.exe, not by the pid exec.Cmd knows.
func TestRollbackKilledCommit(t *testing.T) {
	const pid = 111072 // the launcher's child, never cmd.Process.Pid
	started := time.Now().Add(-40 * time.Second)
	paths := func(gitDir string) (idx, next string) {
		return filepath.Join(gitDir, "index.lock"), filepath.Join(gitDir, NextIndexLockName(pid))
	}

	t.Run("paired locks of the killed run are rolled back", func(t *testing.T) {
		stubKilledCommitAlive(t, nil)
		gitDir := t.TempDir()
		idx, next := paths(gitDir)
		writeLockAt(t, idx, started.Add(100*time.Millisecond))
		writeLockAt(t, next, started.Add(150*time.Millisecond))
		res := rollbackKilledCommitIn(gitDir, started, false)
		if !res.RemovedIndexLock || res.OwnerPID != pid || len(res.RemovedNextIndex) != 1 || res.Err != "" {
			t.Fatalf("want both locks rolled back naming pid %d, got %+v", pid, res)
		}
		if lockFileExists(idx) || lockFileExists(next) {
			t.Fatalf("locks left behind: index.lock=%v next-index=%v", lockFileExists(idx), lockFileExists(next))
		}
	})

	t.Run("a coarse file clock stamping just before start still counts as this run", func(t *testing.T) {
		stubKilledCommitAlive(t, nil)
		gitDir := t.TempDir()
		idx, next := paths(gitDir)
		writeLockAt(t, idx, started.Add(-time.Second))
		writeLockAt(t, next, started.Add(-time.Second))
		if res := rollbackKilledCommitIn(gitDir, started, false); !res.RemovedIndexLock {
			t.Fatalf("a stamp inside the clock slack must still roll back, got %+v", res)
		}
	})

	t.Run("a newer index.lock belongs to a peer and is kept", func(t *testing.T) {
		stubKilledCommitAlive(t, nil)
		gitDir := t.TempDir()
		idx, next := paths(gitDir)
		writeLockAt(t, next, started.Add(100*time.Millisecond))
		writeLockAt(t, idx, started.Add(5*time.Second))
		res := rollbackKilledCommitIn(gitDir, started, false)
		if res.RemovedIndexLock || !res.IndexLockUnpaired || len(res.RemovedNextIndex) != 1 {
			t.Fatalf("want the peer's index.lock kept and only our dead next-index removed, got %+v", res)
		}
		if !lockFileExists(idx) || lockFileExists(next) {
			t.Fatalf("index.lock present=%v (want true), next-index present=%v (want false)", lockFileExists(idx), lockFileExists(next))
		}
	})

	t.Run("residue older than the run proves nothing", func(t *testing.T) {
		stubKilledCommitAlive(t, nil)
		gitDir := t.TempDir()
		idx, next := paths(gitDir)
		writeLockAt(t, idx, started.Add(-2*time.Minute))
		writeLockAt(t, next, started.Add(-time.Minute))
		res := rollbackKilledCommitIn(gitDir, started, false)
		if res.RemovedIndexLock || len(res.RemovedNextIndex) != 0 || !lockFileExists(idx) || !lockFileExists(next) {
			t.Fatalf("a pre-existing orphan is the reapers' call, not the rollback's, got %+v", res)
		}
	})

	t.Run("a live pairing writer is left alone", func(t *testing.T) {
		stubKilledCommitAlive(t, map[int]bool{pid: true})
		gitDir := t.TempDir()
		idx, next := paths(gitDir)
		writeLockAt(t, idx, started.Add(100*time.Millisecond))
		writeLockAt(t, next, started.Add(150*time.Millisecond))
		res := rollbackKilledCommitIn(gitDir, started, false)
		if res.RemovedIndexLock || len(res.RemovedNextIndex) != 0 || !lockFileExists(idx) || !lockFileExists(next) {
			t.Fatalf("a live writer's locks must be kept, got %+v", res)
		}
	})

	t.Run("inside another git's hook the outer index.lock is never touched", func(t *testing.T) {
		stubKilledCommitAlive(t, nil)
		gitDir := t.TempDir()
		idx, next := paths(gitDir)
		writeLockAt(t, idx, started.Add(-time.Second)) // the outer, live git's lock
		writeLockAt(t, next, started.Add(150*time.Millisecond))
		res := rollbackKilledCommitIn(gitDir, started, true)
		if res.RemovedIndexLock || !res.IndexRedirected || !lockFileExists(idx) {
			t.Fatalf("GIT_INDEX_FILE redirect must keep <gitdir>/index.lock, got %+v", res)
		}
		if lockFileExists(next) {
			t.Fatal("our own dead next-index file must still be cleared so it cannot mislead the reapers")
		}
	})
}

// TestRecoverStaleIndexLockStandsDownUnderRedirectedIndex: inside another git's hook
// (GIT_INDEX_FILE inherited) the lock git refused on is <that file>.lock and
// <gitdir>/index.lock is the OUTER git's live lock, so even a thirty-minute-old one — which
// the age reap would otherwise take — must be left alone.
func TestRecoverStaleIndexLockStandsDownUnderRedirectedIndex(t *testing.T) {
	events := captureReapEvents(t)
	gitDir := t.TempDir()
	lock := filepath.Join(gitDir, "index.lock")
	writeLockAt(t, lock, time.Now().Add(-30*time.Minute))
	run := func(ctx context.Context, dir string, args ...string) (string, int, error) {
		if len(args) > 0 && args[len(args)-1] == "index" {
			// --git-path index under an outer normal commit's GIT_INDEX_FILE.
			return filepath.ToSlash(lock) + "\n", 0, nil
		}
		return gitDir + "\n", 0, nil // rev-parse --absolute-git-dir
	}
	if _, _, _, handled := recoverStaleIndexLock(context.Background(), run, "", []string{"commit"}); handled {
		t.Fatal("a redirected index must leave the lock to the classifier, not reap it")
	}
	if !lockFileExists(lock) || len(*events) != 0 {
		t.Fatalf("outer git's index.lock was touched: present=%v events=%v", lockFileExists(lock), *events)
	}
}

// TestReapStaleIndexLockNamedDeadOwner: the in-commit reaper treats a paired
// next-index-<pid>.lock with a dead pid as proof the index.lock is orphaned, reaping it
// (and the partner) after the short frozen window instead of the fifteen-minute one.
func TestReapStaleIndexLockNamedDeadOwner(t *testing.T) {
	const pid = 111072
	stubAlive := func(t *testing.T, alive bool) {
		prev := staleIndexAlive
		staleIndexAlive = func(int) bool { return alive }
		t.Cleanup(func() { staleIndexAlive = prev })
	}
	setup := func(t *testing.T, lockAge, partnerOffset time.Duration) (gitDir, idx, next string) {
		gitDir = t.TempDir()
		idx, next = filepath.Join(gitDir, "index.lock"), filepath.Join(gitDir, NextIndexLockName(pid))
		lockMod := time.Now().Add(-lockAge)
		writeLockAt(t, idx, lockMod)
		writeLockAt(t, next, lockMod.Add(partnerOffset))
		return gitDir, idx, next
	}

	t.Run("incident shape: 2-minute lock, dead paired owner, reaped with partner", func(t *testing.T) {
		stubAlive(t, false)
		gitDir, idx, next := setup(t, 2*time.Minute, time.Second)
		res := reapStaleIndexLock(gitDir, DefaultStaleIndexLockAge)
		if !res.Reaped || !res.ReapedByOwner || res.OwnerPID != pid || !res.OwnerDead {
			t.Fatalf("want an owner-proven reap naming pid %d, got %+v", pid, res)
		}
		if lockFileExists(idx) || lockFileExists(next) || res.PartnerPath != next {
			t.Fatalf("index.lock present=%v next-index present=%v partner=%q", lockFileExists(idx), lockFileExists(next), res.PartnerPath)
		}
	})

	t.Run("live paired owner is never reaped", func(t *testing.T) {
		stubAlive(t, true)
		gitDir, idx, _ := setup(t, 2*time.Minute, time.Second)
		if res := reapStaleIndexLock(gitDir, DefaultStaleIndexLockAge); res.Reaped || res.OwnerDead || !lockFileExists(idx) {
			t.Fatalf("a live partial commit's lock must be kept, got %+v", res)
		}
	})

	t.Run("dead owner not yet frozen the short window is kept", func(t *testing.T) {
		stubAlive(t, false)
		gitDir, idx, _ := setup(t, 20*time.Second, time.Second)
		if res := reapStaleIndexLock(gitDir, DefaultStaleIndexLockAge); res.Reaped || !lockFileExists(idx) {
			t.Fatalf("a lock frozen under %s must be kept, got %+v", DefaultPartialCommitOwnerDeadAge, res)
		}
	})

	t.Run("residue older than the lock names nobody", func(t *testing.T) {
		stubAlive(t, false)
		gitDir, idx, next := setup(t, 2*time.Minute, -time.Minute)
		res := reapStaleIndexLock(gitDir, DefaultStaleIndexLockAge)
		if res.Reaped || res.OwnerPID != 0 || !lockFileExists(idx) || !lockFileExists(next) {
			t.Fatalf("a peer's newer lock must not be pinned on old residue, got %+v", res)
		}
	})
}
