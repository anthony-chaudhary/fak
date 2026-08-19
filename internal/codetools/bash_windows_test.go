//go:build windows

package codetools

import (
	"os/exec"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func TestCancellationTaskkillUsesHiddenWindowPosture(t *testing.T) {
	cmd := exec.Command("taskkill.exe", "/T", "/F", "/PID", "1")
	windowgate.ConfigureBackgroundCommand(cmd)
	const createNoWindow = 0x08000000
	if cmd.SysProcAttr == nil || !cmd.SysProcAttr.HideWindow || cmd.SysProcAttr.CreationFlags&createNoWindow == 0 {
		t.Fatalf("background taskkill posture = %#v, want hidden window with CREATE_NO_WINDOW", cmd.SysProcAttr)
	}
}
