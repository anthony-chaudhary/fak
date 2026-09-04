package workerworktree

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestWorkerLeaseLifecycle(t *testing.T) {
	dir := t.TempDir()
	now := time.Date(2026, 9, 4, 10, 0, 0, 0, time.UTC)
	lease := WorkerLease{
		PID:         12345,
		SessionID:   "sess-42",
		CreatedAt:   now,
		HeartbeatTS: now,
	}

	if err := WriteWorkerLease(dir, lease); err != nil {
		t.Fatalf("WriteWorkerLease failed: %v", err)
	}

	got, err := ReadWorkerLease(dir)
	if err != nil {
		t.Fatalf("ReadWorkerLease failed: %v", err)
	}
	if got.PID != 12345 || got.SessionID != "sess-42" || !got.CreatedAt.Equal(now) || !got.HeartbeatTS.Equal(now) {
		t.Fatalf("ReadWorkerLease got %+v, want %+v", got, lease)
	}

	time.Sleep(10 * time.Millisecond)
	if err := UpdateHeartbeat(dir); err != nil {
		t.Fatalf("UpdateHeartbeat failed: %v", err)
	}

	updated, err := ReadWorkerLease(dir)
	if err != nil {
		t.Fatalf("ReadWorkerLease after update failed: %v", err)
	}
	if !updated.HeartbeatTS.After(now) {
		t.Fatalf("HeartbeatTS %s was not updated after %s", updated.HeartbeatTS, now)
	}
}

func TestMergeTreeRebaseClean(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	rawGit(t, root, "init", "-q", "-b", "main")
	rawGit(t, root, "config", "user.email", "rebase@test")
	rawGit(t, root, "config", "user.name", "rebase")
	rawGit(t, root, "config", "commit.gpgsign", "false")

	// Base commit with f1 and f2
	if err := os.WriteFile(filepath.Join(root, "f1.txt"), []byte("base1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "f2.txt"), []byte("base2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	rawGit(t, root, "add", ".")
	rawGit(t, root, "commit", "-q", "-m", "base")
	baseSHA := strings.TrimSpace(mustGit(t, root, "rev-parse", "HEAD"))

	// Peer moves HEAD by modifying f1
	if err := os.WriteFile(filepath.Join(root, "f1.txt"), []byte("peer1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	rawGit(t, root, "commit", "-q", "-am", "peer commit")
	targetHEAD := strings.TrimSpace(mustGit(t, root, "rev-parse", "HEAD"))

	// Worker committed f2 on top of baseSHA
	rawGit(t, root, "checkout", "-q", baseSHA)
	if err := os.WriteFile(filepath.Join(root, "f2.txt"), []byte("worker2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	rawGit(t, root, "commit", "-q", "-am", "worker commit")
	workerCommit := strings.TrimSpace(mustGit(t, root, "rev-parse", "HEAD"))

	// Restore main
	rawGit(t, root, "checkout", "-q", "main")

	// Rebase worker commit onto targetHEAD in memory
	rebasedTree, err := mergeTreeRebase(root, baseSHA, targetHEAD, workerCommit, nil)
	if err != nil {
		t.Fatalf("mergeTreeRebase failed: %v", err)
	}
	if rebasedTree == "" {
		t.Fatal("mergeTreeRebase returned empty tree")
	}

	// Verify tree content: f1 should be "peer1\n", f2 should be "worker2\n"
	rc, out1 := rawGit(t, root, "cat-file", "-p", rebasedTree+":f1.txt")
	if rc != 0 || out1 != "peer1\n" {
		t.Fatalf("expected f1 in rebased tree to be peer1, got rc=%d out=%q", rc, out1)
	}
	rc, out2 := rawGit(t, root, "cat-file", "-p", rebasedTree+":f2.txt")
	if rc != 0 || out2 != "worker2\n" {
		t.Fatalf("expected f2 in rebased tree to be worker2, got rc=%d out=%q", rc, out2)
	}
}

func TestMergeTreeRebaseConflict(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	rawGit(t, root, "init", "-q", "-b", "main")
	rawGit(t, root, "config", "user.email", "rebase@test")
	rawGit(t, root, "config", "user.name", "rebase")
	rawGit(t, root, "config", "commit.gpgsign", "false")

	if err := os.WriteFile(filepath.Join(root, "conflict.txt"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	rawGit(t, root, "add", ".")
	rawGit(t, root, "commit", "-q", "-m", "base")
	baseSHA := strings.TrimSpace(mustGit(t, root, "rev-parse", "HEAD"))

	if err := os.WriteFile(filepath.Join(root, "conflict.txt"), []byte("peer change\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	rawGit(t, root, "commit", "-q", "-am", "peer commit")
	targetHEAD := strings.TrimSpace(mustGit(t, root, "rev-parse", "HEAD"))

	rawGit(t, root, "checkout", "-q", baseSHA)
	if err := os.WriteFile(filepath.Join(root, "conflict.txt"), []byte("worker change\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	rawGit(t, root, "commit", "-q", "-am", "worker commit")
	workerCommit := strings.TrimSpace(mustGit(t, root, "rev-parse", "HEAD"))

	rawGit(t, root, "checkout", "-q", "main")

	_, err := mergeTreeRebase(root, baseSHA, targetHEAD, workerCommit, nil)
	if err == nil {
		t.Fatal("expected mergeTreeRebase to fail on conflicting edits, got nil error")
	}
}

func TestLandingQueueSerialization(t *testing.T) {
	q := NewLandingQueue()
	root := "/test/repo/path"
	var active atomic.Int32
	var maxActive atomic.Int32
	var wg sync.WaitGroup

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			q.Coordinate(root, func() Result {
				cur := active.Add(1)
				defer active.Add(-1)
				for {
					m := maxActive.Load()
					if cur <= m || maxActive.CompareAndSwap(m, cur) {
						break
					}
				}
				time.Sleep(5 * time.Millisecond)
				return Result{OK: true}
			})
		}()
	}
	wg.Wait()

	if m := maxActive.Load(); m != 1 {
		t.Fatalf("LandingQueue must serialize executions, maximum concurrency was %d", m)
	}
}

func TestReapDeadWorktreeCleansStaleLocks(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	rawGit(t, root, "init", "-q", "-b", "main")
	rawGit(t, root, "config", "user.email", "reap@test")
	rawGit(t, root, "config", "user.name", "reap")
	rawGit(t, root, "config", "commit.gpgsign", "false")

	seedFile := filepath.Join(root, "init.txt")
	if err := os.WriteFile(seedFile, []byte("init\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	rawGit(t, root, "add", "init.txt")
	rawGit(t, root, "commit", "-q", "-m", "init")

	wtRoot := t.TempDir()

	// 1. Prepare an initial worker worktree
	prep := Prepare(root, "crashedlane", "crashedkey", "", wtRoot, nil)
	if !prep.OK {
		t.Fatalf("initial prepare failed: %+v", prep)
	}
	crashedPath := prep.Path

	// 2. Lock the worktree to simulate a crash while locked
	rc, lockOut := rawGit(t, root, "worktree", "lock", crashedPath, "--reason", "simulated crashed worker")
	if rc != 0 {
		t.Fatalf("git worktree lock failed: %s", lockOut)
	}

	// Verify the worktree is locked
	_, listBefore := rawGit(t, root, "worktree", "list")
	if !strings.Contains(listBefore, "locked") {
		t.Fatalf("expected worktree to be locked, list: %s", listBefore)
	}

	// 3. Mark the lease with an inactive PID (99999999) and stale timestamp
	staleTime := time.Now().Add(-30 * time.Minute)
	deadLease := WorkerLease{
		PID:         99999999, // guaranteed inactive PID
		SessionID:   "crashed-session",
		CreatedAt:   staleTime,
		HeartbeatTS: staleTime,
	}
	if err := WriteWorkerLease(crashedPath, deadLease); err != nil {
		t.Fatalf("WriteWorkerLease failed: %v", err)
	}
	_ = writeOwnerStamp(crashedPath, OwnerStamp{
		Schema:    ownerStampSchema,
		PID:       99999999,
		LeaseID:   "crashed-session",
		CreatedAt: staleTime,
	})

	// 4. Invoke Prepare for a new worktree.
	// The automated dead worktree sweep in Prepare must detect the dead/stale worktree,
	// forcibly unlock it, prune it, and successfully initialize the new worktree.
	newRes := Prepare(root, "freshlane", "freshkey", "", wtRoot, nil)
	if !newRes.OK {
		t.Fatalf("Prepare failed after sweeping dead worktree: %+v", newRes)
	}

	// 5. Prove the dead worktree is pruned and no stale lock remains
	_, listAfter := rawGit(t, root, "worktree", "list")
	if strings.Contains(listAfter, crashedPath) {
		t.Fatalf("dead worktree %q still present in git worktree list:\n%s", crashedPath, listAfter)
	}
	if strings.Contains(listAfter, "locked") {
		t.Fatalf("stale lock still present in git worktree list:\n%s", listAfter)
	}

	// 6. Prove the new worktree has a valid lease.json
	newLease, err := ReadWorkerLease(newRes.Path)
	if err != nil {
		t.Fatalf("new worktree missing lease.json: %v", err)
	}
	if newLease.PID <= 0 {
		t.Fatalf("new lease PID = %d, want > 0", newLease.PID)
	}
}
