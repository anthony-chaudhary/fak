//go:build !windows

package wipinventory

import "os/exec"

func configureDispatchHelperCommand(_ *exec.Cmd) {}
