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
	const detachedProcess = uint32(0x00000008)
	if cmd.SysProcAttr.CreationFlags&detachedProcess == 0 {
		t.Fatalf("Codex log helper creation flags %#x omit DETACHED_PROCESS", cmd.SysProcAttr.CreationFlags)
	}
}
