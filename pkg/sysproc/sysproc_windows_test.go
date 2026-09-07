//go:build windows

package sysproc

import (
	"os/exec"
	"testing"
)

func assertCommandConfigured(t *testing.T, cmd *exec.Cmd) {
	t.Helper()
	if cmd.SysProcAttr == nil {
		t.Fatal("SysProcAttr is nil on Windows")
	}
	if !cmd.SysProcAttr.HideWindow {
		t.Error("HideWindow is false, want true")
	}
	if cmd.SysProcAttr.CreationFlags&CreateNoWindow == 0 {
		t.Errorf("CreationFlags=%#x missing CreateNoWindow (%#x)", cmd.SysProcAttr.CreationFlags, CreateNoWindow)
	}
}

func TestConfigureBackground(t *testing.T) {
	cmd := exec.Command("cmd", "/c", "exit", "0")
	ConfigureBackground(cmd)

	if cmd.SysProcAttr == nil {
		t.Fatal("SysProcAttr is nil on Windows")
	}
	if !cmd.SysProcAttr.HideWindow {
		t.Error("HideWindow is false, want true")
	}
	if cmd.SysProcAttr.CreationFlags&CreateNoWindow == 0 {
		t.Errorf("CreationFlags=%#x missing CreateNoWindow (%#x)", cmd.SysProcAttr.CreationFlags, CreateNoWindow)
	}
}

func TestConfigureDetached(t *testing.T) {
	cmd := exec.Command("cmd", "/c", "exit", "0")
	// First configure as background to prove CreateNoWindow gets cleared
	ConfigureBackground(cmd)
	ConfigureDetached(cmd)

	if cmd.SysProcAttr == nil {
		t.Fatal("SysProcAttr is nil on Windows")
	}
	if !cmd.SysProcAttr.HideWindow {
		t.Error("HideWindow is false, want true")
	}
	if cmd.SysProcAttr.CreationFlags&CreateNoWindow != 0 {
		t.Errorf("CreationFlags=%#x should not have CreateNoWindow set", cmd.SysProcAttr.CreationFlags)
	}
	if cmd.SysProcAttr.CreationFlags&DetachedProcess == 0 {
		t.Errorf("CreationFlags=%#x missing DetachedProcess (%#x)", cmd.SysProcAttr.CreationFlags, DetachedProcess)
	}
}

func TestConfigureProcessGroup(t *testing.T) {
	cmd := exec.Command("cmd", "/c", "exit", "0")
	ConfigureProcessGroup(cmd)

	if cmd.SysProcAttr == nil {
		t.Fatal("SysProcAttr is nil on Windows")
	}
	if !cmd.SysProcAttr.HideWindow {
		t.Error("HideWindow is false, want true")
	}
	if cmd.SysProcAttr.CreationFlags&CreateNoWindow == 0 {
		t.Errorf("CreationFlags=%#x missing CreateNoWindow (%#x)", cmd.SysProcAttr.CreationFlags, CreateNoWindow)
	}
	if cmd.SysProcAttr.CreationFlags&CreateNewProcessGroup == 0 {
		t.Errorf("CreationFlags=%#x missing CreateNewProcessGroup (%#x)", cmd.SysProcAttr.CreationFlags, CreateNewProcessGroup)
	}
}
