//go:build !windows

package armbench

import "os/exec"

func configureDispatchHelperCommand(_ *exec.Cmd) {}
