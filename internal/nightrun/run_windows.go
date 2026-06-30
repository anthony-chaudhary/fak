//go:build windows

package nightrun

import (
	"os/exec"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// configureProcGroup makes a timed-out / cancelled task kill the WHOLE process
// tree on Windows, not just the launched cmd.exe. A collection command is
// typically `cmd /c go run ./cmd/<bench>` (or a prebuilt binary), where the real
// worker is a grandchild; Go's default Cancel only TerminateProcess-es the direct
// child, orphaning the worker — which keeps holding the GPU/box and the output
// pipe (blocking CombinedOutput past the deadline). We override Cancel to
// `taskkill /T /F /PID <pid>`, which terminates the child AND its entire
// descendant tree, so a timed-out task frees the box and the loop moves on. This
// matters because detectGPU can mark a Windows NVIDIA host a feasible CUDA box, so
// Windows is not merely hypothetical as a collection host. cmd.WaitDelay (set in
// execTask) remains the portable backstop if taskkill is unavailable.
func configureProcGroup(cmd *exec.Cmd) {
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
