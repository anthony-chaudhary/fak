//go:build windows

package devindex

import "testing"

func TestConfigureGraphCommandDetachesConsoleTool(t *testing.T) {
	cmd := graphCommand("go", "version")
	if cmd.SysProcAttr == nil || cmd.SysProcAttr.CreationFlags == 0 {
		t.Fatal("graph command is not configured as a detached console helper")
	}
}
