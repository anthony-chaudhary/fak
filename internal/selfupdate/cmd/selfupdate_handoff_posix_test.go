//go:build !windows

package selfupdatecmd

import (
	"context"
	"os"
	"testing"
)

func TestSelfUpdateSuccessorPOSIXProcessGroup(t *testing.T) {
	cmd := selfUpdateSuccessorCommand(context.Background(), os.Args[0], nil, "session", "revision")
	if cmd.SysProcAttr == nil || !cmd.SysProcAttr.Setpgid {
		t.Fatalf("SysProcAttr = %+v, want independent POSIX process group", cmd.SysProcAttr)
	}
}
