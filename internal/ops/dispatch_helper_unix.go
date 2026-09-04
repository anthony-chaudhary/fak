//go:build !windows

package ops

import "os/exec"

func configureDispatchHelperCommand(_ *exec.Cmd) {}
