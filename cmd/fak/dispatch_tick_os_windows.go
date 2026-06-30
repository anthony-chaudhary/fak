//go:build windows

package main

import (
	"os/exec"
	"strconv"
	"strings"
)

func configureDispatchSpawn(cmd *exec.Cmd) {
	configureDispatchHelperCommand(cmd)
}

func dispatchPIDAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	cmd := exec.Command("tasklist", "/FI", "PID eq "+strconv.Itoa(pid), "/NH")
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return false
	}
	return strings.Contains(string(out), strconv.Itoa(pid))
}
