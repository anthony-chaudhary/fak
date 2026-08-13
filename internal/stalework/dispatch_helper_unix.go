//go:build !windows

package stalework

import "os/exec"

func configureDispatchHelperCommand(_ *exec.Cmd) {}
