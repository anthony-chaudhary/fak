//go:build windows

package main

import (
	"os/exec"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func TestConfigureDispatchHelperCommandAllocatesNoConsole(t *testing.T) {
	cmd := exec.Command("cmd.exe", "/c", "exit", "0")
	configureDispatchHelperCommand(cmd)

	if cmd.SysProcAttr == nil {
		t.Fatal("dispatch helper has no Windows process attributes")
	}
	const (
		detachedProcess = 0x00000008
		createNoWindow  = 0x08000000
	)
	flags := cmd.SysProcAttr.CreationFlags
	if flags&detachedProcess == 0 {
		t.Fatalf("dispatch helper flags %#x omit DETACHED_PROCESS; child would allocate a console", flags)
	}
	if flags&createNoWindow != 0 {
		t.Fatalf("dispatch helper flags %#x retain mutually exclusive CREATE_NO_WINDOW", flags)
	}
}

func TestConfigureDispatchWorkerSpawnKeepsCodexDescendantsHidden(t *testing.T) {
	t.Run("codex inherits one hidden console", func(t *testing.T) {
		cmd := exec.Command("cmd.exe", "/c", "exit", "0")
		configureDispatchSpawn(cmd)
		configureDispatchWorkerConsole(cmd, "codex")
		if cmd.SysProcAttr == nil || !cmd.SysProcAttr.HideWindow {
			t.Fatalf("Codex worker attributes = %#v, want hidden window", cmd.SysProcAttr)
		}
		flags := cmd.SysProcAttr.CreationFlags
		if flags&windowgate.CreateNoWindow == 0 || flags&windowgate.DetachedProcess != 0 {
			t.Fatalf("Codex worker flags = %#x, want CREATE_NO_WINDOW without DETACHED_PROCESS", flags)
		}
	})

	t.Run("other backends remain detached", func(t *testing.T) {
		cmd := exec.Command("cmd.exe", "/c", "exit", "0")
		configureDispatchSpawn(cmd)
		configureDispatchWorkerConsole(cmd, "claude")
		flags := cmd.SysProcAttr.CreationFlags
		if flags&windowgate.DetachedProcess == 0 || flags&windowgate.CreateNoWindow != 0 {
			t.Fatalf("non-Codex worker flags = %#x, want DETACHED_PROCESS without CREATE_NO_WINDOW", flags)
		}
	})
}
