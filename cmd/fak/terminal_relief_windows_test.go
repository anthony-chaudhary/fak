//go:build windows

package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"syscall"
	"testing"
)

func TestTerminalReliefSpawnModesSuppressConsoleWindows(t *testing.T) {
	background := newTerminalReliefBackgroundCommand("powershell.exe", "-NoProfile")
	if background.SysProcAttr == nil || !background.SysProcAttr.HideWindow {
		t.Fatal("terminal-relief background helper would show a window")
	}
	if background.SysProcAttr.CreationFlags&0x08000000 == 0 {
		t.Fatalf("terminal-relief background helper flags %#x omit CREATE_NO_WINDOW", background.SysProcAttr.CreationFlags)
	}

	detached := newTerminalReliefDetachedCommand([]string{"fak.exe", "info"})
	if detached.SysProcAttr == nil || !detached.SysProcAttr.HideWindow {
		t.Fatal("terminal-relief dashboard relaunch would show a window")
	}
	const detachedProcess = 0x00000008
	if detached.SysProcAttr.CreationFlags&detachedProcess == 0 {
		t.Fatalf("terminal-relief dashboard flags %#x omit DETACHED_PROCESS", detached.SysProcAttr.CreationFlags)
	}
	if detached.SysProcAttr.CreationFlags&0x08000000 != 0 {
		t.Fatalf("terminal-relief dashboard flags %#x combine mutually exclusive CREATE_NO_WINDOW and DETACHED_PROCESS", detached.SysProcAttr.CreationFlags)
	}
	if detached.SysProcAttr.CreationFlags&syscall.CREATE_NEW_PROCESS_GROUP == 0 {
		t.Fatalf("terminal-relief dashboard flags %#x omit CREATE_NEW_PROCESS_GROUP", detached.SysProcAttr.CreationFlags)
	}
}

func TestTerminalReliefBelowThresholdDoesNotAct(t *testing.T) {
	oldGather, oldLaunch, oldStop := gatherTerminalReliefSnapshotFn, launchTerminalReliefCommandFn, stopTerminalReliefHostFn
	defer func() {
		gatherTerminalReliefSnapshotFn = oldGather
		launchTerminalReliefCommandFn = oldLaunch
		stopTerminalReliefHostFn = oldStop
	}()
	gatherTerminalReliefSnapshotFn = func() (terminalReliefSnapshot, error) {
		return terminalReliefSnapshot{PID: 7, Handles: 9999, Threads: 499}, nil
	}
	launchTerminalReliefCommandFn = func([]string) error { t.Fatal("launch called"); return nil }
	stopTerminalReliefHostFn = func(int) error { t.Fatal("stop called"); return nil }
	var out, errout bytes.Buffer
	if rc := runTerminalRelief(&out, &errout, []string{"--apply", "--json", "--state", filepath.Join(t.TempDir(), "state.json")}); rc != 0 {
		t.Fatalf("rc=%d err=%s", rc, errout.String())
	}
	var got struct {
		Verdict string `json:"verdict"`
		Apply   bool   `json:"apply"`
	}
	if json.Unmarshal(out.Bytes(), &got) != nil || got.Verdict != "BELOW_THRESHOLD" || got.Apply {
		t.Fatalf("out=%s", out.String())
	}
}
func TestTerminalReliefPersistentSafeHostRelaunchesThenStops(t *testing.T) {
	oldGather, oldLaunch, oldStop := gatherTerminalReliefSnapshotFn, launchTerminalReliefCommandFn, stopTerminalReliefHostFn
	defer func() {
		gatherTerminalReliefSnapshotFn = oldGather
		launchTerminalReliefCommandFn = oldLaunch
		stopTerminalReliefHostFn = oldStop
	}()
	gatherTerminalReliefSnapshotFn = func() (terminalReliefSnapshot, error) {
		return terminalReliefSnapshot{PID: 7, Handles: 12000, Threads: 600, Processes: []terminalReliefProcess{{PID: 8, ParentPID: 7, Name: "pwsh.exe"}, {PID: 9, ParentPID: 8, Name: "fak.exe", CommandLine: `C:\Users\u\bin\fak.exe info --gateway-url http://127.0.0.1:1 --interval 2s`}}}, nil
	}
	var order []string
	launchTerminalReliefCommandFn = func(argv []string) error { order = append(order, "launch:"+argv[1]); return nil }
	stopTerminalReliefHostFn = func(pid int) error { order = append(order, "stop"); return nil }
	state := filepath.Join(t.TempDir(), "state.json")
	for i := 0; i < 3; i++ {
		var out, errout bytes.Buffer
		rc := runTerminalRelief(&out, &errout, []string{"--apply", "--json", "--state", state, "--cooldown", "0s"})
		if rc != 0 {
			t.Fatalf("tick=%d rc=%d err=%s", i, rc, errout.String())
		}
	}
	if len(order) != 2 || order[0] != "launch:info" || order[1] != "stop" {
		t.Fatalf("order=%v", order)
	}
}
func TestTerminalReliefUnsafeDescendantAbstains(t *testing.T) {
	old := gatherTerminalReliefSnapshotFn
	defer func() { gatherTerminalReliefSnapshotFn = old }()
	gatherTerminalReliefSnapshotFn = func() (terminalReliefSnapshot, error) {
		return terminalReliefSnapshot{PID: 7, Handles: 12000, Threads: 600, Processes: []terminalReliefProcess{{PID: 8, ParentPID: 7, Name: "vim.exe"}}}, nil
	}
	var out, errout bytes.Buffer
	if rc := runTerminalRelief(&out, &errout, []string{"--apply", "--json", "--state", filepath.Join(t.TempDir(), "state.json"), "--consecutive", "1"}); rc != 3 {
		t.Fatalf("rc=%d out=%s err=%s", rc, out.String(), errout.String())
	}
}
