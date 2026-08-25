//go:build windows

package procguard

import (
	"errors"
	"os"
	"syscall"
	"testing"
)

func TestCollectCommitSnapshotOwnProcess(t *testing.T) {
	s, supported, detail := CollectCommitSnapshot(os.Getpid())
	if !supported {
		t.Fatal("Windows commit accounting must be supported")
	}
	if detail != "" {
		t.Fatalf("detail=%q", detail)
	}
	if s.SystemCommitLimit == 0 || s.SystemCommitBytes == 0 || s.SystemCommitBytes >= s.SystemCommitLimit {
		t.Fatalf("system snapshot=%+v", s)
	}
	found := false
	for _, p := range s.Processes {
		if p.PID == os.Getpid() && p.CommitBytes > 0 {
			found = true
		}
	}
	if !found {
		t.Fatalf("own PID missing or has zero commit: %+v", s.Processes)
	}
}

func TestProcessExitedDuringSnapshot(t *testing.T) {
	if !processExitedDuringSnapshot(syscall.Errno(87)) {
		t.Fatal("a vanished Windows PID must not abort child resource monitoring")
	}
	if processExitedDuringSnapshot(syscall.ERROR_ACCESS_DENIED) {
		t.Fatal("access-denied telemetry must remain fail-closed")
	}
	if processExitedDuringSnapshot(errors.New("other monitor failure")) {
		t.Fatal("unknown telemetry failures must remain fail-closed")
	}
}
