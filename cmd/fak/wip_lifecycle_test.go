package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/wiplifecycle"
)

func TestWIPLifecycleCommandBracketsClearOutWithOneReceipt(t *testing.T) {
	repo := initWipAdmitRepo(t)
	if err := os.WriteFile(filepath.Join(repo, "before.go"), []byte("package before\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runWIPLifecycle([]string{"begin", "--kind", "clear-out", "--root", repo}, &out, &errOut); code != 0 {
		t.Fatalf("begin code=%d out=%s err=%s", code, out.String(), errOut.String())
	}
	var begun wiplifecycle.Receipt
	if err := json.Unmarshal(out.Bytes(), &begun); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(repo, "before.go")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "after.go"), []byte("package after\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errOut.Reset()
	if code := runWIPLifecycle([]string{"end", "--id", begun.OperationID, "--root", repo}, &out, &errOut); code != 0 {
		t.Fatalf("end code=%d out=%s err=%s", code, out.String(), errOut.String())
	}
	var finished wiplifecycle.Receipt
	if err := json.Unmarshal(out.Bytes(), &finished); err != nil {
		t.Fatal(err)
	}
	if finished.OperationID != begun.OperationID || !finished.Before.Known || !finished.After.Known {
		t.Fatalf("receipt not linked: %#v", finished)
	}
}

func TestAutomaticWIPLifecycleDoesNotBlockRecoveryWhenCaptureFails(t *testing.T) {
	var errOut bytes.Buffer
	finish := beginAutomaticWIPLifecycle(filepath.Join(t.TempDir(), "not-a-repository"), "crash-recovery", &errOut)
	finish()
	if got := errOut.String(); !bytes.Contains([]byte(got), []byte("WIP_LIFECYCLE_CAPTURE_FAILED phase=before")) {
		t.Fatalf("typed non-blocking failure missing: %s", got)
	}
}

func TestWorkerLifecycleMutationHooksStayWired(t *testing.T) {
	body := readEntrypoint(t, "worktree_worker.go")
	for _, want := range []string{
		`beginAutomaticWIPLifecycleWithGit(repoRoot, "worker-reap"`,
		`beginAutomaticWIPLifecycle(repoRoot, "crash-recovery"`,
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("automatic lifecycle hook missing: %s", want)
		}
	}
}

func TestRunWIPLifecycleListRendersDurableHistory(t *testing.T) {
	repo := initWipAdmitRepo(t)
	var out, errOut bytes.Buffer
	if code := runWIPLifecycle([]string{"begin", "--root", repo, "--kind", "checkpoint-reap", "--id", "history-1"}, &out, &errOut); code != 0 {
		t.Fatalf("begin rc=%d stderr=%s", code, errOut.String())
	}
	out.Reset()
	errOut.Reset()
	if code := runWIPLifecycle([]string{"end", "--root", repo, "--id", "history-1"}, &out, &errOut); code != 0 {
		t.Fatalf("end rc=%d stderr=%s", code, errOut.String())
	}
	out.Reset()
	errOut.Reset()
	if code := runWIPLifecycle([]string{"list", "--root", repo}, &out, &errOut); code != 0 {
		t.Fatalf("list rc=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"FINISHED", "checkpoint-reap", "history-1", "receipt.json"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("list missing %q:\n%s", want, out.String())
		}
	}
}
