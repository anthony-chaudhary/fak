package gateway

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

const durableWriterHelperEnv = "FAK_DURABLE_WRITER_HELPER"

func provisionDurableWriterJournal(t *testing.T, path string) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatalf("provision journal: %v", err)
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		t.Fatalf("sync provisioned journal: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close provisioned journal: %v", err)
	}
}

func openDurableWriterJournal(t *testing.T, path string) *DurableControlIngress {
	t.Helper()
	ingress, err := OpenDurableControlIngress(DurableControlIngressOptions{
		JournalPath:     path,
		MaxJournalBytes: 1 << 20,
		MaxPending:      8,
	})
	if err != nil {
		t.Fatalf("open durable control ingress: %v", err)
	}
	return ingress
}

func postDurableWriterDirective(t *testing.T, ingress http.Handler, id string) (int, ControlReceipt) {
	t.Helper()
	directive := ControlDirective{
		ID:         id,
		Target:     "mission-writer-ownership",
		Action:     "pause",
		Generation: 1,
		Payload:    json.RawMessage(`{"source":"writer-process-test"}`),
	}
	body, err := json.Marshal(directive)
	if err != nil {
		t.Fatalf("marshal directive: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/fak/control/directives", bytes.NewReader(body))
	w := httptest.NewRecorder()
	ingress.ServeHTTP(w, req)
	var receipt ControlReceipt
	if w.Body.Len() != 0 {
		if err := json.Unmarshal(w.Body.Bytes(), &receipt); err != nil {
			t.Fatalf("decode status %d response %q: %v", w.Code, w.Body.String(), err)
		}
	}
	return w.Code, receipt
}

func requireOwnedJournalError(t *testing.T, err error) {
	t.Helper()
	if !errors.Is(err, errControlJournalOwned) {
		t.Fatalf("contender error = %v, want errors.Is(err, errControlJournalOwned)", err)
	}
}

// TestDurableWriterProcessHelper is executed only as a subprocess. It keeps an
// ingress live until its parent kills the process, giving the parent a real OS
// lock lifetime instead of an in-process approximation.
func TestDurableWriterProcessHelper(t *testing.T) {
	path := os.Getenv(durableWriterHelperEnv)
	if path == "" {
		return
	}
	ingress, err := OpenDurableControlIngress(DurableControlIngressOptions{
		JournalPath:     path,
		MaxJournalBytes: 1 << 20,
		MaxPending:      8,
	})
	if err != nil {
		t.Fatalf("helper open: %v", err)
	}
	defer ingress.Close()
	fmt.Println("READY")
	for {
		time.Sleep(time.Hour)
	}
}

func startDurableWriterProcess(t *testing.T, path string) *exec.Cmd {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestDurableWriterProcessHelper$", "-test.count=1")
	cmd.Env = append(os.Environ(), durableWriterHelperEnv+"="+path)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("helper stdout pipe: %v", err)
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Fatalf("start helper: %v", err)
	}
	t.Cleanup(func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	})

	ready := make(chan string, 1)
	go func() {
		scanner := bufio.NewScanner(stdout)
		if scanner.Scan() {
			ready <- scanner.Text()
			return
		}
		ready <- "helper exited before READY: " + stderr.String()
	}()
	select {
	case line := <-ready:
		if strings.TrimSpace(line) != "READY" {
			t.Fatalf("helper readiness = %q", line)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timed out waiting for helper ownership")
	}
	return cmd
}

func stopDurableWriterProcess(t *testing.T, cmd *exec.Cmd) {
	t.Helper()
	if err := cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
		t.Fatalf("kill helper: %v", err)
	}
	if err := cmd.Wait(); err == nil {
		t.Fatal("killed helper exited successfully; expected process termination")
	}
	cmd.Process = nil
}

func TestDurableControlIngressRejectsConcurrentWriterInSameProcess(t *testing.T) {
	path := filepath.Join(t.TempDir(), "control.journal")
	provisionDurableWriterJournal(t, path)
	owner := openDurableWriterJournal(t, path)
	defer owner.Close()

	contender, err := OpenDurableControlIngress(DurableControlIngressOptions{JournalPath: path})
	if err == nil {
		_ = contender.Close()
		t.Fatal("second ingress acquired a journal already owned in this process")
	}
	requireOwnedJournalError(t, err)
}

func TestDurableControlIngressProcessDeathReleasesWriterOwnership(t *testing.T) {
	path := filepath.Join(t.TempDir(), "control.journal")
	provisionDurableWriterJournal(t, path)
	helper := startDurableWriterProcess(t, path)

	contender, err := OpenDurableControlIngress(DurableControlIngressOptions{JournalPath: path})
	if err == nil {
		_ = contender.Close()
		t.Fatal("parent acquired a journal owned by a live child process")
	}
	requireOwnedJournalError(t, err)

	stopDurableWriterProcess(t, helper)
	deadline := time.Now().Add(5 * time.Second)
	for {
		owner, openErr := OpenDurableControlIngress(DurableControlIngressOptions{JournalPath: path})
		if openErr == nil {
			if err := owner.Close(); err != nil {
				t.Fatalf("close owner acquired after process death: %v", err)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("writer ownership remained held after OS process death: %v", openErr)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestDurableControlIngressAliasesCannotBypassWriterOwnership(t *testing.T) {
	tests := []struct {
		name string
		link func(oldname, newname string) error
	}{
		{name: "hardlink", link: os.Link},
		{name: "symlink", link: os.Symlink},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "control.journal")
			alias := filepath.Join(dir, "control-alias.journal")
			provisionDurableWriterJournal(t, path)
			if err := tt.link(path, alias); err != nil {
				if tt.name == "symlink" && (runtime.GOOS == "windows" || errors.Is(err, fs.ErrPermission)) {
					t.Skipf("symlink creation is unavailable on this host: %v", err)
				}
				t.Fatalf("create %s: %v", tt.name, err)
			}

			owner := openDurableWriterJournal(t, path)
			defer owner.Close()
			contender, err := OpenDurableControlIngress(DurableControlIngressOptions{JournalPath: alias})
			if err == nil {
				_ = contender.Close()
				t.Fatalf("%s alias acquired simultaneous writer ownership", tt.name)
			}
			requireOwnedJournalError(t, err)
		})
	}
}

func TestDurableControlIngressHardlinkAliasCannotBypassCrossProcessOwnership(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "control.journal")
	alias := filepath.Join(dir, "control-hardlink.journal")
	provisionDurableWriterJournal(t, path)
	if err := os.Link(path, alias); err != nil {
		t.Fatalf("create hardlink: %v", err)
	}
	helper := startDurableWriterProcess(t, path)
	defer stopDurableWriterProcess(t, helper)

	contender, err := OpenDurableControlIngress(DurableControlIngressOptions{JournalPath: alias})
	if err == nil {
		_ = contender.Close()
		t.Fatal("hardlink alias acquired writer ownership held by another process")
	}
	requireOwnedJournalError(t, err)
}

type durableWriterSyncFailureFile struct {
	writes int
	syncs  int
}

func (f *durableWriterSyncFailureFile) Write(p []byte) (int, error) {
	f.writes++
	return len(p), nil
}

func (f *durableWriterSyncFailureFile) Sync() error {
	f.syncs++
	if f.syncs == 1 {
		return nil // Recovery fence succeeds; the attempted accepted-frame Sync fails.
	}
	return errors.New("independent injected journal Sync failure")
}

func (*durableWriterSyncFailureFile) Close() error { return nil }

func TestDurableControlIngressFailedSyncSameIDRetryReturnsStoredUnknown(t *testing.T) {
	path := filepath.Join(t.TempDir(), "control.journal")
	provisionDurableWriterJournal(t, path)
	fake := &durableWriterSyncFailureFile{}
	originalOpen := openControlIngressJournal
	openControlIngressJournal = func(string, int, fs.FileMode) (controlIngressJournalFile, error) {
		return fake, nil
	}
	t.Cleanup(func() { openControlIngressJournal = originalOpen })

	ingress, err := OpenDurableControlIngress(DurableControlIngressOptions{
		JournalPath:     path,
		MaxJournalBytes: 1 << 20,
		MaxPending:      8,
	})
	if err != nil {
		t.Fatalf("open injected-Sync ingress: %v", err)
	}
	t.Cleanup(func() { _ = ingress.Close() })

	status, first := postDurableWriterDirective(t, ingress, "same-id-after-sync-failure")
	if status != http.StatusServiceUnavailable {
		t.Fatalf("initial failed-Sync status = %d, want %d", status, http.StatusServiceUnavailable)
	}
	if first.State != "unknown" || first.Reason != "journal_indeterminate" || first.Digest == "" {
		t.Fatalf("initial failed-Sync receipt = %+v, want stored unknown with reason and digest", first)
	}
	writesAfterFailure, syncsAfterFailure := fake.writes, fake.syncs

	status, retry := postDurableWriterDirective(t, ingress, "same-id-after-sync-failure")
	if status != http.StatusServiceUnavailable {
		t.Fatalf("same-ID retry status = %d, want %d", status, http.StatusServiceUnavailable)
	}
	if retry.State != first.State || retry.Reason != first.Reason || retry.Digest != first.Digest {
		t.Fatalf("same-ID retry receipt = %+v, want stored original %+v", retry, first)
	}
	if fake.writes != writesAfterFailure || fake.syncs != syncsAfterFailure {
		t.Fatalf("same-ID retry retried poisoned journal I/O: writes %d->%d, syncs %d->%d", writesAfterFailure, fake.writes, syncsAfterFailure, fake.syncs)
	}
}

func TestDurableControlIngressPathReplacementFailsClosed(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "control.journal")
	displaced := filepath.Join(dir, "control-displaced.journal")
	provisionDurableWriterJournal(t, path)
	ingress := openDurableWriterJournal(t, path)
	defer ingress.Close()

	if err := os.Rename(path, displaced); err != nil {
		if runtime.GOOS == "windows" {
			t.Skipf("Windows filesystem denies replacement while the journal handle is open: %v", err)
		}
		t.Fatalf("displace owned journal: %v", err)
	}
	provisionDurableWriterJournal(t, path)
	status, _ := postDurableWriterDirective(t, ingress, "replaced-path")
	if status != http.StatusServiceUnavailable {
		t.Fatalf("POST after path replacement status = %d, want %d (and never %d)", status, http.StatusServiceUnavailable, http.StatusAccepted)
	}
}

func TestDurableControlIngressSymlinkRepointFailsClosed(t *testing.T) {
	dir := t.TempDir()
	targetA := filepath.Join(dir, "control-a.journal")
	targetB := filepath.Join(dir, "control-b.journal")
	link := filepath.Join(dir, "control-current.journal")
	provisionDurableWriterJournal(t, targetA)
	provisionDurableWriterJournal(t, targetB)
	if err := os.Symlink(targetA, link); err != nil {
		if runtime.GOOS == "windows" || errors.Is(err, fs.ErrPermission) {
			t.Skipf("symlink creation is unavailable on this host: %v", err)
		}
		t.Fatalf("create journal symlink: %v", err)
	}
	ingress := openDurableWriterJournal(t, link)
	defer ingress.Close()

	if err := os.Remove(link); err != nil {
		t.Fatalf("remove original journal symlink: %v", err)
	}
	if err := os.Symlink(targetB, link); err != nil {
		t.Fatalf("repoint journal symlink: %v", err)
	}
	status, _ := postDurableWriterDirective(t, ingress, "repointed-symlink")
	if status != http.StatusServiceUnavailable {
		t.Fatalf("POST after symlink repoint status = %d, want %d (and never %d)", status, http.StatusServiceUnavailable, http.StatusAccepted)
	}
}
