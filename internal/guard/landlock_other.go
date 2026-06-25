//go:build !linux || (linux && !amd64 && !arm64)

package guard

import (
	"fmt"
	"os"
	"os/exec"
)

// LandlockTrampoline on a platform without the Landlock hook-floor is a transparent
// passthrough: it logs that the floor is unavailable here and runs the real agent
// UNrestricted. This is the same fail-open posture as an old Linux kernel — the floor is
// Linux-only (kernel 5.13+) opt-in defense-in-depth, never a cross-platform guarantee, and
// must never block a spawn on a host that cannot provide it.
//
// It exists so cmd/fak builds and the trampoline verb is wired identically on every GOOS;
// in practice the parent only inserts the trampoline on linux/amd64|arm64, so this path is
// reached only if the hidden verb is invoked directly on another platform.
func LandlockTrampoline(args []string) error {
	_, agentArgv, ok := SplitTrampolineArgs(args)
	if !ok {
		return fmt.Errorf("guard: landlock trampoline: malformed args (need <spec> -- <agent argv>)")
	}
	fmt.Fprintln(os.Stderr, "fak guard: landlock hook-floor not available on this platform; running agent unrestricted")
	if len(agentArgv) == 0 {
		return fmt.Errorf("guard: landlock trampoline: empty agent argv")
	}
	cmd := exec.Command(agentArgv[0], agentArgv[1:]...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	cmd.Env = os.Environ()
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			os.Exit(ee.ExitCode())
		}
		return err
	}
	os.Exit(0)
	return nil
}
