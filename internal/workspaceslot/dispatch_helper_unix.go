//go:build !windows

package workspaceslot

import "os/exec"

func configureDispatchHelperCommand(_ *exec.Cmd) {}
