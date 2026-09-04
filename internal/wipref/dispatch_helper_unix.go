//go:build !windows

package wipref

import "os/exec"

func configureDispatchHelperCommand(_ *exec.Cmd) {}
