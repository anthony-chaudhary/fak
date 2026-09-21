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

// TestConfigureDurableChild asserts the durability seam emits
// CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP with HideWindow, and critically
// does NOT set DETACHED_PROCESS (which would kill the child with the launcher,
// fak#13468).
func TestConfigureDurableChild(t *testing.T) {
	cmd := exec.Command("cmd", "/c", "exit", "0")
	// First configure as detached to prove ConfigureDurableChild clears it.
	ConfigureDetached(cmd)
	ConfigureDurableChild(cmd)

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
	if cmd.SysProcAttr.CreationFlags&DetachedProcess != 0 {
		t.Errorf("CreationFlags=%#x must NOT set DetachedProcess (%#x): a detached child dies with its launcher", cmd.SysProcAttr.CreationFlags, DetachedProcess)
	}
}

func TestConfigureDurableChildNilSafety(t *testing.T) {
	ConfigureDurableChild(nil) // must not panic
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
