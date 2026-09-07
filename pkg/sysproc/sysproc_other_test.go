//go:build !windows

package sysproc

import (
	"os/exec"
	"testing"
)

func assertCommandConfigured(t *testing.T, cmd *exec.Cmd) {
	t.Helper()
	// On non-Windows platforms, Command does not set SysProcAttr by default.
}

func TestConfigureBackground(t *testing.T) {
	cmd := exec.Command("true")
	ConfigureBackground(cmd)
	// No-op on non-Windows; no crash.
}

func TestConfigureDetached(t *testing.T) {
	cmd := exec.Command("true")
	ConfigureDetached(cmd)
	if cmd.SysProcAttr == nil {
		t.Fatal("SysProcAttr is nil on POSIX")
	}
	if !cmd.SysProcAttr.Setsid {
		t.Error("Setsid is false, want true")
	}
}

func TestConfigureProcessGroup(t *testing.T) {
	cmd := exec.Command("true")
	ConfigureProcessGroup(cmd)
	if cmd.SysProcAttr == nil {
		t.Fatal("SysProcAttr is nil on POSIX")
	}
	if !cmd.SysProcAttr.Setpgid {
		t.Error("Setpgid is false, want true")
	}
}
