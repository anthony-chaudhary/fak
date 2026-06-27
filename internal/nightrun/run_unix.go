//go:build !windows

package nightrun

import (
	"os/exec"
	"syscall"
)

// configureProcGroup puts the spawned shell in its own process group and replaces
// the context-cancel hook so a timeout kills the WHOLE group, not just the direct
// child. A collection command is typically `go run ./cmd/<bench>`, where the real
// worker is a compiled grandchild; killing only `go run` would leave that worker
// running (and holding the output pipe open, blocking the read). Killing the group
// (-pgid) reaps the worker too, so a timed-out task frees the box and the loop.
func configureProcGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		// Negative pid => signal the process group led by the child (Setpgid above
		// makes the child's pgid == its pid). SIGKILL is unblockable, so a wedged
		// worker cannot ignore it.
		if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); err != nil {
			// Fall back to killing just the child if the group send failed.
			return cmd.Process.Kill()
		}
		return nil
	}
}
