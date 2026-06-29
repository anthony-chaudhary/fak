//go:build !windows

package main

import (
	"context"
	"testing"
)

// TestConfigureProcTreeSetsOwnProcessGroup witnesses the Unix-specific half of the
// tree kill: the worker is placed in its OWN process group (Setpgid), which is what
// makes the Cancel hook's `kill(-pgid)` reach the whole descendant tree rather than
// only the backend shim. Without the own-group move the negative-pid signal has no
// group to hit and grandchildren survive a timed-out worker.
func TestConfigureProcTreeSetsOwnProcessGroup(t *testing.T) {
	cmd := newLaunchCmd(context.Background(), []string{"true"}, "", map[string]string{})
	if cmd.SysProcAttr == nil || !cmd.SysProcAttr.Setpgid {
		t.Fatal("worker cmd must set Setpgid so a timeout can kill the whole process group")
	}
}
