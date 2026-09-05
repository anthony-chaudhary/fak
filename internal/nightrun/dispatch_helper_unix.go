//go:build !windows

package nightrun

import "os/exec"

func configureDispatchHelperCommand(_ *exec.Cmd) {}
