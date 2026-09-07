//go:build windows

package main

import (
	"context"
	"syscall"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func TestConfigureProcTreeUsesWorkerCommandPosture(t *testing.T) {
	cmd := newLaunchCmd(context.Background(), []string{"cmd.exe", "/c", "echo ok"}, "", map[string]string{})
	if cmd.SysProcAttr == nil || !cmd.SysProcAttr.HideWindow || cmd.SysProcAttr.CreationFlags&windowgate.CreateNoWindow == 0 {
		t.Fatalf("worker cmd posture = %#v, want hidden window with CREATE_NO_WINDOW", cmd.SysProcAttr)
	}
	if cmd.SysProcAttr.CreationFlags&syscall.CREATE_NEW_PROCESS_GROUP == 0 {
		t.Fatalf("worker cmd posture = %#v, want CREATE_NEW_PROCESS_GROUP set", cmd.SysProcAttr)
	}
}
