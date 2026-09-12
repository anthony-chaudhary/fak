//go:build windows

package main

import (
	"os/exec"
	"testing"
)

func TestDoctorCodexLogHelperSuppressesBackgroundWindow(t *testing.T) {
	cmd := exec.Command("python", "--version")
	configureDispatchHelperCommand(cmd)
	if cmd.SysProcAttr == nil || !cmd.SysProcAttr.HideWindow {
		t.Fatalf("Codex log helper is not window-suppressed: %#v", cmd.SysProcAttr)
	}
	// Helpers need a hidden console for descendants to inherit. Detaching the
	// helper instead lets ordinary console descendants create visible windows.
	const createNoWindow = uint32(0x08000000)
	if cmd.SysProcAttr.CreationFlags&createNoWindow == 0 {
		t.Fatalf("Codex log helper creation flags %#x omit CREATE_NO_WINDOW", cmd.SysProcAttr.CreationFlags)
	}
	const detachedProcess = uint32(0x00000008)
	if cmd.SysProcAttr.CreationFlags&detachedProcess != 0 {
		t.Fatalf("Codex log helper must retain its hidden console for descendants: flags %#x", cmd.SysProcAttr.CreationFlags)
	}
}
