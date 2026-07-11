package safecommit

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// isIndexLockContention is NARROWER than isGitLockContention: only the index lock
// is auto-reaped, so a ref-lock / packed-refs failure — also contention — must not
// enter the reap path.
func TestIsIndexLockContention(t *testing.T) {
	cases := []struct {
		name string
		out  string
		want bool
	}{
		{"index lock fatal", indexLockFatal, true},
		{"ref lock", "error: cannot lock ref 'refs/heads/main': Unable to create 'C:/work/fak/.git/refs/heads/main.lock': File exists.", false},
		{"packed refs", "fatal: Unable to create 'C:/work/fak/.git/packed-refs.lock': File exists", false},
		{"hook refusal", "FILE_ADMISSION: docs/tmp-scratch.md is a one-off operational artifact", false},
		{"empty", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isIndexLockContention(tc.out); got != tc.want {
				t.Fatalf("isIndexLockContention(%q) = %v, want %v", tc.out, got, tc.want)
			}
		})
	}
}

// reapStaleIndexLock removes an index.lock only when it is present AND older than
// the threshold; a fresh or absent lock is left exactly as found.
func TestReapStaleIndexLock(t *testing.T) {
	t.Run("stale lock is reaped", func(t *testing.T) {
		dir := t.TempDir()
		lock := filepath.Join(dir, "index.lock")
		writeSafecommitFile(t, lock, "")
		old := time.Now().Add(-30 * time.Minute) // a crash artifact, not a live hold
		if err := os.Chtimes(lock, old, old); err != nil {
			t.Fatal(err)
		}
		res := reapStaleIndexLock(dir, DefaultStaleIndexLockAge)
		if !res.Reaped || !res.Present {
			t.Fatalf("a stale lock must reap, got %+v", res)
		}
		if res.AgeSeconds < int64((DefaultStaleIndexLockAge).Seconds()) {
			t.Fatalf("reported age %ds must exceed the %s threshold", res.AgeSeconds, DefaultStaleIndexLockAge)
		}
		if _, err := os.Stat(lock); !os.IsNotExist(err) {
			t.Fatalf("the stale lock file must be removed, stat err = %v", err)
		}
	})

	t.Run("fresh lock is left untouched", func(t *testing.T) {
		dir := t.TempDir()
		lock := filepath.Join(dir, "index.lock")
		writeSafecommitFile(t, lock, "") // mtime = now: a live git may hold it
		res := reapStaleIndexLock(dir, DefaultStaleIndexLockAge)
		if res.Reaped || !res.Present {
			t.Fatalf("a fresh lock must be present but not reaped, got %+v", res)
		}
		if _, err := os.Stat(lock); err != nil {
			t.Fatalf("a fresh lock must remain on disk, stat err = %v", err)
		}
	})

	t.Run("absent lock is a no-op", func(t *testing.T) {
		dir := t.TempDir()
		res := reapStaleIndexLock(dir, DefaultStaleIndexLockAge)
		if res.Reaped || res.Present {
			t.Fatalf("an absent lock must reap nothing, got %+v", res)
		}
	})

	t.Run("zero threshold never reaps", func(t *testing.T) {
		dir := t.TempDir()
		lock := filepath.Join(dir, "index.lock")
		writeSafecommitFile(t, lock, "")
		old := time.Now().Add(-30 * time.Minute)
		if err := os.Chtimes(lock, old, old); err != nil {
			t.Fatal(err)
		}
		if res := reapStaleIndexLock(dir, 0); res.Reaped {
			t.Fatalf("a non-positive threshold must disable reaping, got %+v", res)
		}
	})
}

// A crashed writer's abandoned index.lock outlives the in-place riding retries,
// then is reaped and the commit lands — the automatic form of the manual `rm`
// git's crash text demands (#3915). The fake git refuses `add` exactly while the
// real index.lock file exists, mirroring git's own behavior, so the reap is what
// unblocks it.
func TestStaleIndexLock_reapedThenCommits(t *testing.T) {
	events := captureReapEvents(t)
	swallowContentionSleep(t)
	gitDir := t.TempDir()
	lock := filepath.Join(gitDir, "index.lock")
	writeSafecommitFile(t, lock, "")
	old := time.Now().Add(-30 * time.Minute)
	if err := os.Chtimes(lock, old, old); err != nil {
		t.Fatal(err)
	}

	g := &fakeGit{reply: onTrunkBase()}
	g.reply["rev-parse --absolute-git-dir"] = reply{out: gitDir + "\n", code: 0}
	addCalls := 0
	run := func(ctx context.Context, dir string, args ...string) (string, int, error) {
		if len(args) > 0 && args[0] == "add" {
			addCalls++
			if _, err := os.Stat(lock); err == nil {
				return indexLockFatal, 128, nil // git refuses while index.lock exists
			}
		}
		return g.run(ctx, dir, args...)
	}

	res, err := CommitWith(context.Background(), run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != "" || !res.Committed || !res.Verified {
		t.Fatalf("a stale index.lock must be reaped and the commit land, got %+v", res)
	}
	if _, err := os.Stat(lock); !os.IsNotExist(err) {
		t.Fatalf("the stale index.lock must be removed, stat err = %v", err)
	}
	if addCalls != lockContentionAttempts+1 {
		t.Fatalf("add attempts = %d, want %d (rode contention, reaped, retried once)", addCalls, lockContentionAttempts+1)
	}
	if len(*events) != 1 || !strings.Contains((*events)[0], "INDEX_LOCK_REAPED") {
		t.Fatalf("want exactly one INDEX_LOCK_REAPED event, got %v", *events)
	}
}

// A FRESH index.lock (a live git may hold it) is NOT reaped: the commit fails
// LOCK_BUSY carrying a precise "another git process is active" message, never
// git's generic crash text, and the lock file is left on disk.
func TestFreshIndexLock_reportsLiveHolderNotCrashText(t *testing.T) {
	events := captureReapEvents(t)
	swallowContentionSleep(t)
	gitDir := t.TempDir()
	lock := filepath.Join(gitDir, "index.lock")
	writeSafecommitFile(t, lock, "") // mtime = now: fresh, a live holder

	g := &fakeGit{reply: onTrunkBase()}
	g.reply["rev-parse --absolute-git-dir"] = reply{out: gitDir + "\n", code: 0}
	run := func(ctx context.Context, dir string, args ...string) (string, int, error) {
		if len(args) > 0 && args[0] == "add" {
			return indexLockFatal, 128, nil
		}
		return g.run(ctx, dir, args...)
	}

	res, err := CommitWith(context.Background(), run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != ReasonLockBusy {
		t.Fatalf("Reason = %q, want %q", res.Reason, ReasonLockBusy)
	}
	if !strings.Contains(res.Detail, "another git process is active") {
		t.Fatalf("Detail must name a live holder, got %q", res.Detail)
	}
	if strings.Contains(strings.ToLower(res.Detail), "crashed") ||
		strings.Contains(res.Detail, "remove the file manually") {
		t.Fatalf("Detail must replace git's crash text, got %q", res.Detail)
	}
	if res.Committed {
		t.Fatalf("nothing may commit while a live holder has index.lock: %+v", res)
	}
	if _, err := os.Stat(lock); err != nil {
		t.Fatalf("a fresh index.lock must NOT be reaped, stat err = %v", err)
	}
	if len(*events) != 0 {
		t.Fatalf("no reap event may fire for a fresh lock, got %v", *events)
	}
}
