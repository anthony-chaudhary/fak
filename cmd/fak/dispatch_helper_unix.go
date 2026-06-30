//go:build !windows

package main

import "os/exec"

func configureDispatchHelperCommand(_ *exec.Cmd) {}
