//go:build !windows

package codedebt

import "os/exec"

func configureDispatchHelperCommand(_ *exec.Cmd) {}
