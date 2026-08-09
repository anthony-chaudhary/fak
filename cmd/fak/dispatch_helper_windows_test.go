//go:build windows

package main

import (
	"os/exec"
	"testing"
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
