//go:build !windows

package main

import (
	"os/exec"
	"syscall"

	"github.com/anthony-chaudhary/fak/internal/dispatchaudit"
)

func configureDispatchSpawn(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
}

func configureDispatchWorkerConsole(*exec.Cmd, string) {}

// dispatchPIDAlive is dispatchaudit.ProcessAlive — the one liveness probe. The tick's
// scan loops (livescan, preflight, witness) and the dispatch audit must never disagree
// about whether a worker is still running; this file used to carry a byte-identical
// private copy of the signal-0 probe.
func dispatchPIDAlive(pid int) bool { return dispatchaudit.ProcessAlive(pid) }
