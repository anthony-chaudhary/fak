//go:build windows

package selfupdatecmd

import (
	"context"
	"os"
	"syscall"
	"testing"
)

func TestSelfUpdateSuccessorWindowsProcessGroup(t *testing.T) {
	cmd := selfUpdateSuccessorCommand(context.Background(), os.Args[0], nil, "session", "revision")
	if cmd.SysProcAttr == nil || cmd.SysProcAttr.CreationFlags&syscall.CREATE_NEW_PROCESS_GROUP == 0 {
		t.Fatalf("SysProcAttr = %+v, want CREATE_NEW_PROCESS_GROUP", cmd.SysProcAttr)
	}
}
