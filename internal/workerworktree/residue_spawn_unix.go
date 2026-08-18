//go:build !windows

package workerworktree

import "os/exec"

func configureDispatchHelperCommand(_ *exec.Cmd) {}
