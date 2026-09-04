//go:build windows

package procguard

import (
	"errors"
	"os"
	"syscall"
	"testing"
)

func TestWindowsCommitProcessAccessIsLeastPrivilege(t *testing.T) {
	if windowsCommitProcessAccess != processQueryLimitedInformation {
		t.Fatalf("access mask=%#x want %#x", windowsCommitProcessAccess, processQueryLimitedInformation)
	}
	if windowsCommitProcessAccess&processVMRead != 0 {
		t.Fatalf("access mask %#x includes PROCESS_VM_READ", windowsCommitProcessAccess)
	}
}

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

func TestCollectMemorySnapshotOwnProcessPhysicalMemory(t *testing.T) {
	s, supported, detail := CollectMemorySnapshot(os.Getpid())
	if !supported {
		t.Fatal("Windows memory accounting must be supported")
	}
	if detail != "" {
		t.Fatalf("detail=%q", detail)
	}
	if s.HostPhysicalBytes == 0 {
		t.Fatal("HostPhysicalBytes must be > 0")
	}
	if s.HostPhysicalAvailableBytes == 0 {
		t.Fatal("HostPhysicalAvailableBytes must be > 0")
	}
	if s.HostPhysicalAvailableBytes > s.HostPhysicalBytes {
		t.Fatalf("HostPhysicalAvailableBytes (%d) > HostPhysicalBytes (%d)", s.HostPhysicalAvailableBytes, s.HostPhysicalBytes)
	}
}

func TestWindowsCommitOwnedPIDsRejectsStalePPIDEdges(t *testing.T) {
	parentPID := func(pid int) *int { return &pid }
	rows := []Proc{
		{PID: 100, Start: "2026-08-31T12:00:00Z"},
		{PID: 101, PPID: parentPID(100), Start: "2026-08-31T11:59:59Z"},
		{PID: 102, PPID: parentPID(101), Start: "2026-08-31T12:00:01Z"},
		{PID: 103, PPID: parentPID(100), Start: "2026-08-31T12:00:01Z"},
		{PID: 104, PPID: parentPID(103), Start: "2026-08-31T12:00:02Z"},
	}
	byPID := make(map[int]Proc, len(rows))
	children := make(map[int][]Proc)
	for _, row := range rows {
		byPID[row.PID] = row
		if row.PPID != nil {
			children[*row.PPID] = append(children[*row.PPID], row)
		}
	}

	owned := windowsCommitOwnedPIDs(100, byPID, children)
	for _, pid := range []int{100, 103, 104} {
		if !owned[pid] {
			t.Errorf("genuine descendant pid %d excluded: %v", pid, owned)
		}
	}
	for _, pid := range []int{101, 102} {
		if owned[pid] {
			t.Errorf("stale PPID edge admitted pid %d: %v", pid, owned)
		}
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
