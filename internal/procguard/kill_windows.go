//go:build windows

package procguard

import "errors"

// killSignal is unused on Windows (KillPID shells taskkill /T /F directly); it
// exists only so the POSIX SIGKILL path compiles cross-platform.
func killSignal(int) error {
	return errors.New("killSignal not used on windows; KillPID uses taskkill")
}
