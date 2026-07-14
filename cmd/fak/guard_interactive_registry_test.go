package main

import (
	"os"
	"runtime"
	"testing"
)

func TestGuardOwnsInteractiveTerminal(t *testing.T) {
	keys := []string{"FAK_DISPATCH_ID", "FAK_HEADLESS", "CI", "WT_SESSION", "WT_WINDOW", "TERM_PROGRAM", "TERM", "SSH_TTY"}
	old := map[string]string{}
	had := map[string]bool{}
	for _, k := range keys {
		old[k], had[k] = os.LookupEnv(k)
		_ = os.Unsetenv(k)
	}
	t.Cleanup(func() {
		for _, k := range keys {
			if had[k] {
				_ = os.Setenv(k, old[k])
			} else {
				_ = os.Unsetenv(k)
			}
		}
	})

	switch runtime.GOOS {
	case "windows":
		_ = os.Setenv("WT_SESSION", "interactive")
	case "darwin":
		_ = os.Setenv("TERM_PROGRAM", "Apple_Terminal")
	default:
		_ = os.Setenv("TERM", "xterm-256color")
	}
	if !guardOwnsInteractiveTerminal() {
		t.Fatal("attached operator terminal was not classified interactive")
	}
	_ = os.Setenv("FAK_DISPATCH_ID", "worker-1")
	if guardOwnsInteractiveTerminal() {
		t.Fatal("dispatcher-owned session classified interactive")
	}
}
