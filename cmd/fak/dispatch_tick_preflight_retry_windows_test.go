//go:build windows

package main

import (
	"bytes"
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func TestDispatchHostProbePowerShellKeepsUsablePipesOnWindows(t *testing.T) {
	cmd := windowgate.CommandContext(context.Background(), "powershell", "-NoProfile", "-NonInteractive", "-Command", `Write-Output '{"ok":true}'`)
	if cmd.SysProcAttr == nil || cmd.SysProcAttr.CreationFlags&windowgate.CreateNoWindow == 0 {
		t.Fatalf("host probe flags=%#v, want CREATE_NO_WINDOW", cmd.SysProcAttr)
	}
	if cmd.SysProcAttr.CreationFlags&windowgate.DetachedProcess != 0 {
		t.Fatalf("host probe flags=%#x include DETACHED_PROCESS, which drops redirected PowerShell output", cmd.SysProcAttr.CreationFlags)
	}
	out, err := cmd.Output()
	if err != nil || !bytes.Contains(out, []byte(`{"ok":true}`)) {
		t.Fatalf("out=%q err=%v", out, err)
	}
}
