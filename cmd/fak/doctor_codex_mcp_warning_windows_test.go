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
	const createNoWindow = uint32(0x08000000)
	if cmd.SysProcAttr.CreationFlags&createNoWindow == 0 {
		t.Fatalf("Codex log helper creation flags %#x omit CREATE_NO_WINDOW", cmd.SysProcAttr.CreationFlags)
	}
}
