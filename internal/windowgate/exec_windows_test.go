//go:build windows

package windowgate

import (
	"os/exec"
	"testing"
)

func TestConfigureBackgroundCommandSetsWindowsNoWindow(t *testing.T) {
	cmd := exec.Command("cmd", "/c", "exit", "0")
	ConfigureBackgroundCommand(cmd)
	if cmd.SysProcAttr == nil {
		t.Fatal("SysProcAttr is nil")
	}
	if !cmd.SysProcAttr.HideWindow {
		t.Fatal("HideWindow = false, want true")
	}
	if cmd.SysProcAttr.CreationFlags&CreateNoWindow == 0 {
		t.Fatalf("CreationFlags=%#x missing CREATE_NO_WINDOW", cmd.SysProcAttr.CreationFlags)
	}
}
