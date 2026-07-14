//go:build windows

package main

import (
	"io"
	"testing"
	"time"

	"golang.org/x/sys/windows/svc"
)

func TestWindowsServiceHandlerReportsRunningAndStops(t *testing.T) {
	old := serviceTick
	serviceTick = func(io.Writer, io.Writer) int { time.Sleep(5 * time.Millisecond); return 0 }
	t.Cleanup(func() { serviceTick = old })
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
