//go:build windows

package main

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hostresurrect"
	"golang.org/x/sys/windows/svc"
	"golang.org/x/sys/windows/svc/mgr"
)

func TestWindowsServiceHandlerReportsRunningAndStops(t *testing.T) {
	oldCrash, oldResume := windowsControlCrashTick, windowsControlResumeTick
	windowsControlCrashTick = func(io.Writer, io.Writer, string) int { return 0 }
	windowsControlResumeTick = func(io.Writer, io.Writer) int { return 0 }
	t.Cleanup(func() { windowsControlCrashTick, windowsControlResumeTick = oldCrash, oldResume })
	changes := make(chan svc.ChangeRequest, 1)
	statuses := make(chan svc.Status, 8)
	done := make(chan struct{})
	go func() { fakWindowsService{io.Discard, io.Discard}.Execute(nil, changes, statuses); close(done) }()
	seenRunning := false
	deadline := time.After(time.Second)
	for !seenRunning {
		select {
		case st := <-statuses:
			seenRunning = st.State == svc.Running
		case <-deadline:
			t.Fatal("no Running status")
		}
	}
	changes <- svc.ChangeRequest{Cmd: svc.Stop}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("service did not stop")
	}
}
func TestWindowsServiceDryRunNamesLeastPrivilegeSCMUnit(t *testing.T) {
	r, rc := windowsServiceAction("install", io.Discard, io.Discard, true)
	if rc != 0 || r.Manager != "windows-scm" || r.Unit != windowsGuardServiceName {
		t.Fatalf("r=%+v rc=%d", r, rc)
	}
}

func TestWindowsServiceWitnessDryRunIsNonDestructive(t *testing.T) {
	r, rc := windowsServiceAction("witness", io.Discard, io.Discard, true)
	if rc != 0 || r.Manager != "windows-scm" || r.Unit != windowsGuardServiceName || !r.StateKept || r.PIDBefore != 0 || r.PIDAfter != 0 {
		t.Fatalf("r=%+v rc=%d", r, rc)
	}
}

func TestWindowsControlCrashTickUsesMachineInteractiveRegistry(t *testing.T) {
	state := t.TempDir()
	registry := filepath.Join(state, "registry")
	if rc := windowsControlCrashTick(io.Discard, io.Discard, state); rc != 0 {
		t.Fatalf("tick rc=%d", rc)
	}
	if _, err := os.Stat(filepath.Join(registry, hostresurrect.CohortFileName)); err != nil {
		t.Fatalf("cohort not written to interactive registry: %v", err)
	}
}

func TestWindowsStagedServiceExecutableUsesProgramData(t *testing.T) {
	old := windowsServiceProgramData
	windowsServiceProgramData = func() string { return `C:\ProgramData` }
	t.Cleanup(func() { windowsServiceProgramData = old })
	want := filepath.Join(`C:\ProgramData`, "fak", "bin", "fak.exe")
	if got := windowsStagedServiceExecutable(); got != want {
		t.Fatalf("path=%q want=%q", got, want)
	}
	cfg := windowsServiceConfig(want)
	if cfg.StartType != mgr.StartAutomatic || cfg.ServiceStartName != `NT AUTHORITY\LocalService` || !strings.Contains(cfg.BinaryPathName, want) {
		t.Fatalf("config=%+v", cfg)
	}
}
