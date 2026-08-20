//go:build windows

package main

import (
	"os/exec"

	"github.com/anthony-chaudhary/fak/internal/dispatchaudit"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func configureDispatchSpawn(cmd *exec.Cmd) {
	configureDispatchHelperCommand(cmd)
}

// configureDispatchWorkerConsole keeps Codex's console-subsystem descendants on
// one inherited hidden console. A detached Codex has no console to inherit, so
// each PowerShell, Node, or stdio MCP child can allocate a visible one instead.
// Other backends retain the lower-cost no-console posture from #3597.
func configureDispatchWorkerConsole(cmd *exec.Cmd, backend string) {
	if cmd == nil || !dispatchWorkerNeedsHiddenConsole(backend) {
		return
	}
	if cmd.SysProcAttr != nil {
		cmd.SysProcAttr.CreationFlags &^= windowgate.DetachedProcess
	}
	windowgate.ConfigureBackgroundCommand(cmd)
}

// dispatchPIDAlive reports whether a pid is currently running — WITHOUT spawning
// a process. It previously shelled `tasklist /FI` once per call, and the dispatch
// tick calls it 50-80 times per tick across its scan loops (livescan, preflight,
// witness). On a multi-session box that per-pid `tasklist` fan-out is a dominant
// process-spawn storm — each cold `tasklist` start is a burst of soft faults,
// context switches and syscalls that stalls the whole machine while CPU/RAM read
// low (see `fak stallscan`). Replacing it with the same
// OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)+GetExitCodeProcess syscall pair
// that internal/safecommit and internal/dispatchaudit already use turns each of
// those 50-80 spawns per tick into a cheap in-process syscall.
//
// OpenProcess succeeds for a live pid and fails once it has exited. A still-open
// handle to a not-yet-reaped zombie reports its exit code, so we also check
// GetExitCodeProcess: STILL_ACTIVE (259) means running. Any error resolving the
// pid is treated as "not alive" — a pid we cannot confirm live must not keep a
// worker slot or lock wedged.
// It is dispatchaudit.ProcessAlive: that package owns the syscall pair (and the
// zombie/unreadable-exit-code judgement), and this file used to carry a byte-identical
// private copy of it down to its own lazy kernel32 handle.
func dispatchPIDAlive(pid int) bool { return dispatchaudit.ProcessAlive(pid) }
