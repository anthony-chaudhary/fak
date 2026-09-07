//go:build windows

package codetools

import (
	"context"
	"os/exec"
	"strings"
	"syscall"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func TestConfigureProcessTreeUsesWorkerCommandPosture(t *testing.T) {
	cmd := exec.Command("cmd.exe", "/c", "echo ok")
	configureProcessTree(cmd)
	const createNoWindow = 0x08000000
	if cmd.SysProcAttr == nil || !cmd.SysProcAttr.HideWindow || cmd.SysProcAttr.CreationFlags&createNoWindow == 0 {
		t.Fatalf("configureProcessTree posture = %#v, want hidden window with CREATE_NO_WINDOW", cmd.SysProcAttr)
	}
	if cmd.SysProcAttr.CreationFlags&syscall.CREATE_NEW_PROCESS_GROUP == 0 {
		t.Fatalf("configureProcessTree posture = %#v, want CREATE_NEW_PROCESS_GROUP set", cmd.SysProcAttr)
	}
}

func TestCancellationTaskkillUsesHiddenWindowPosture(t *testing.T) {
	cmd := exec.Command("taskkill.exe", "/T", "/F", "/PID", "1")
	windowgate.ConfigureBackgroundCommand(cmd)
	const createNoWindow = 0x08000000
	if cmd.SysProcAttr == nil || !cmd.SysProcAttr.HideWindow || cmd.SysProcAttr.CreationFlags&createNoWindow == 0 {
		t.Fatalf("background taskkill posture = %#v, want hidden window with CREATE_NO_WINDOW", cmd.SysProcAttr)
	}
}

func TestBashWindowsMissingCwdAvoidsError267(t *testing.T) {
	ts, _ := newTestToolset(t)
	out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: "echo ok", Cwd: "vanished_subdir"}))
	if bad {
		t.Fatalf("bash with missing cwd must not fail with Win32 error 267: %s", out)
	}
	got := decodeResult(t, out)
	if !strings.Contains(got["stdout"].(string), "ok") {
		t.Fatalf("stdout=%v, want ok", got["stdout"])
	}
}
