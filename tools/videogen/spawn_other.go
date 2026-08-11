//go:build !windows

package main

import "os/exec"

func configureDispatchSpawn(_ *exec.Cmd) {}
