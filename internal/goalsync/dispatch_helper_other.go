//go:build !windows

package goalsync

import "os/exec"

func configureDispatchHelperCommand(_ *exec.Cmd) {}
