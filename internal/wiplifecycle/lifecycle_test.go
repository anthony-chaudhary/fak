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

func TestListKeepsFinishedHistoryAfterObservedArtifactsDisappear(t *testing.T) {
	repo := initRepo(t)
	started := time.Unix(1700000000, 0)
	first, err := Begin(repo, "checkpoint-reap", "reap-1", started)
	if err != nil {
		t.Fatal(err)
	}
	finished, err := Finish(repo, first.OperationID, started.Add(time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(repo, "base.txt")); err != nil {
		t.Fatal(err)
	}
	second, err := Begin(repo, "worker-reap", "reap-2", started.Add(2*time.Minute))
	if err != nil {
		t.Fatal(err)
	}

	got, err := List(repo)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("List()=%#v, want two receipts", got)
	}
	if got[0].OperationID != second.OperationID || got[1].OperationID != finished.OperationID {
		t.Fatalf("List() order=%#v, want newest-first", got)
	}
	if got[1].FinishedAt == "" || got[1].ReceiptPath == "" {
		t.Fatalf("finished history lost completion evidence: %#v", got[1])
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

func TestListWithDiagnosticsRetainsValidHistoryBesideCorruptReceipts(t *testing.T) {
	repo := initRepo(t)
	started := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	validID := "20260901T120000.000000000Z-aabbccddeeff"
	if _, err := Begin(repo, "clear-out-wip", validID, started); err != nil {
		t.Fatal(err)
	}

	store := filepath.Join(repo, ".git", storeName)
	missingID := "20260901T120100.000000000Z-112233445566"
	malformedID := "20260901T120200.000000000Z-66778899aabb"
	if err := os.MkdirAll(filepath.Join(store, missingID), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(store, malformedID), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(store, malformedID, receiptFile), []byte("{"), 0o644); err != nil {
		t.Fatal(err)
	}

	result, err := ListWithDiagnostics(repo)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Receipts) != 1 || result.Receipts[0].OperationID != validID {
		t.Fatalf("valid history not retained: %#v", result.Receipts)
	}
	if len(result.Diagnostics) != 2 {
		t.Fatalf("diagnostics=%#v, want missing and malformed", result.Diagnostics)
	}
	if got, want := result.Diagnostics[0].Code, "MISSING_RECEIPT"; got != want {
		t.Fatalf("first diagnostic code=%q, want %q", got, want)
	}
	if got, want := result.Diagnostics[1].Code, "MALFORMED_RECEIPT"; got != want {
		t.Fatalf("second diagnostic code=%q, want %q", got, want)
	}
	if _, err := List(repo); err == nil || !strings.Contains(err.Error(), missingID) {
		t.Fatalf("strict List error=%v, want malformed evidence refusal", err)
	}
	if _, err := Finish(repo, malformedID, started.Add(time.Minute)); err == nil {
		t.Fatal("Finish accepted malformed lifecycle evidence")
	}
	if _, err := os.Stat(filepath.Join(store, missingID)); err != nil {
		t.Fatalf("listing altered missing receipt directory: %v", err)
	}
	body, err := os.ReadFile(filepath.Join(store, malformedID, receiptFile))
	if err != nil || string(body) != "{" {
		t.Fatalf("listing repaired malformed receipt: body=%q err=%v", body, err)
	}
}

func TestBeginCommitsReceiptDirectoryAtomically(t *testing.T) {
	repo := initRepo(t)
	id := "20260901T130000.000000000Z-aabbccddeeff"
	if _, err := Begin(repo, "clear-out-wip", id, time.Now()); err != nil {
		t.Fatal(err)
	}
	operation := filepath.Join(repo, ".git", storeName, id)
	if _, err := Read(filepath.Join(operation, receiptFile)); err != nil {
		t.Fatalf("committed receipt unreadable: %v", err)
	}
	entries, err := os.ReadDir(filepath.Dir(operation))
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), ".operation-") {
			t.Fatalf("staging directory leaked after atomic commit: %s", entry.Name())
		}
	}
}
