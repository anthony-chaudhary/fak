package wiplifecycle

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestBeginFinishCaptureOrderedSnapshotsWithSharedIdentity(t *testing.T) {
	repo := initRepo(t)
	beforePath := filepath.Join(repo, "before.go")
	if err := os.WriteFile(beforePath, []byte("package before\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	started := time.Unix(1700000000, 0)
	receipt, err := Begin(repo, "clear-out", "operation-1", started)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(beforePath); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "after.go"), []byte("package after\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	finished, err := Finish(repo, receipt.OperationID, started.Add(time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if finished.OperationID != receipt.OperationID || finished.Kind != "clear-out" {
		t.Fatalf("identity changed: %#v", finished)
	}
	if !finished.Before.Known || !finished.After.Known || finished.Before.Artifact == finished.After.Artifact {
		t.Fatalf("captures not linked: %#v", finished)
	}
	if finished.StartedAt >= finished.FinishedAt {
		t.Fatalf("capture ordering missing: %#v", finished)
	}
	before, err := os.ReadFile(filepath.FromSlash(finished.Before.Artifact))
	if err != nil || !strings.Contains(string(before), `"before.go"`) || strings.Contains(string(before), `"after.go"`) {
		t.Fatalf("bad before snapshot err=%v body=%s", err, before)
	}
	after, err := os.ReadFile(filepath.FromSlash(finished.After.Artifact))
	if err != nil || strings.Contains(string(after), `"before.go"`) || !strings.Contains(string(after), `"after.go"`) {
		t.Fatalf("bad after snapshot err=%v body=%s", err, after)
	}
}

func TestBeginPersistsCaptureFailureAsUnknown(t *testing.T) {
	repo := initRepo(t)
	if err := os.WriteFile(filepath.Join(repo, ".git", "index"), []byte("invalid index"), 0o644); err != nil {
		t.Fatal(err)
	}
	receipt, err := Begin(repo, "crash-recovery", "operation-unknown", time.Now())
	if err != nil {
		t.Fatalf("capture failure should not prevent receipt persistence: %v", err)
	}
	if receipt.Before.Known || receipt.Before.Error == "" || receipt.Before.Artifact == "" {
		t.Fatalf("capture failure was conflated with known zero: %#v", receipt.Before)
	}
	if _, err := os.Stat(filepath.FromSlash(receipt.Before.Artifact)); err != nil {
		t.Fatalf("failed observation artifact missing: %v", err)
	}
}

func TestBeginRejectsUnpersistableReceiptStore(t *testing.T) {
	repo := initRepo(t)
	gitDir := filepath.Join(repo, ".git", "fak-wip-lifecycle")
	if err := os.WriteFile(gitDir, []byte("not a directory"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Begin(repo, "crash-recovery", "operation-2", time.Now()); err == nil {
		t.Fatal("expected receipt persistence failure")
	}
}

func initRepo(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()
	for _, args := range [][]string{{"init", "-q"}, {"config", "user.email", "test@example.invalid"}, {"config", "user.name", "Test"}} {
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}
	if err := os.WriteFile(filepath.Join(repo, "base.txt"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{{"add", "base.txt"}, {"commit", "-qm", "base"}} {
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}
	return repo
}
