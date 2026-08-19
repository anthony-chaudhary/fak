//go:build windows

package accounts

import (
	"os/exec"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func TestRefreshTreeKillUsesHiddenWindowPosture(t *testing.T) {
	cmd := exec.Command("taskkill", "/PID", "1", "/T", "/F")
	windowgate.ConfigureBackgroundCommand(cmd)
	attr := cmd.SysProcAttr
	const createNoWindow = 0x08000000
	if attr == nil || !attr.HideWindow || attr.CreationFlags&createNoWindow == 0 {
		t.Fatalf("background taskkill posture = %#v, want hidden window with CREATE_NO_WINDOW", attr)
	}
}
