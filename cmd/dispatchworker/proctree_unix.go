//go:build !windows

package main

import (
	"os/exec"
	"syscall"
)

// configureProcTree puts the spawned worker in its own process group and replaces
// the context-cancel hook so a timed-out launch kills the WHOLE group, not just the
// direct child. A dispatch worker is a full agentic `claude -p` / `opencode run`
// session that, in turn, spawns the real work as grandchildren — `go test`, `grep`,
// a `make ci`, a model load. Go's default CommandContext cancel only signals the
// direct child (the backend shim); killing only it would leave those grandchildren
// running, still holding the box, the commit lock, and the GPU lease — so one
// wedged worker blocks every other agent on the shared trunk. Killing the group
// (-pgid) reaps the descendants too, so a timed-out worker frees the box and the
// supervisor loop moves on.
//
// This mirrors internal/nightrun.configureProcGroup; the worker is launched
// UNATTENDED by the supervisor (`dos loop --enact` / the watchdog canary), so its
// stdin is not a controlling TTY and the own-group move raises no SIGTTIN.
func configureProcTree(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		// Negative pid => signal the process group led by the child (Setpgid above
		// makes the child's pgid == its pid). SIGKILL is unblockable, so a wedged
		// session cannot ignore it.
		if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); err != nil {
			// Fall back to killing just the child if the group send failed.
			return cmd.Process.Kill()
		}
		return nil
	}
}
