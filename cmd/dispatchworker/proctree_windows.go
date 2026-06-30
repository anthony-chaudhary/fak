//go:build windows

package main

import (
	"os/exec"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// configureProcTree makes a timed-out launch kill the WHOLE process tree on
// Windows, not just the launched backend shim. A dispatch worker is a full agentic
// `claude -p` / `opencode run` session that spawns the real work as grandchildren —
// `go test`, `grep`, a `make ci`, a model load. Go's default CommandContext cancel
// only TerminateProcess-es the direct child; killing only it would orphan those
// grandchildren, which keep holding the box, the commit lock, and the GPU lease —
// so one wedged worker blocks every other agent on the shared trunk. We override
// Cancel to `taskkill /T /F /PID <pid>`, which terminates the child AND its entire
// descendant tree, so a timed-out worker frees the box and the supervisor loop
// moves on. cmd.WaitDelay (set in newLaunchCmd) remains the portable backstop if
// taskkill is unavailable. This mirrors internal/nightrun.configureProcGroup.
func configureProcTree(cmd *exec.Cmd) {
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		pid := strconv.Itoa(cmd.Process.Pid)
		kill := exec.Command("taskkill", "/T", "/F", "/PID", pid)
		windowgate.ConfigureBackgroundCommand(kill)
		if err := kill.Run(); err != nil {
			// taskkill unavailable / process already gone — fall back to the child.
			return cmd.Process.Kill()
		}
		return nil
	}
}
