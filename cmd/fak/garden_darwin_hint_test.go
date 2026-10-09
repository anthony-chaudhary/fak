package main

import (
	"context"
	"os/exec"
	"runtime"
	"testing"
	"time"
)

// fak-test:runtime fast est=20ms
func TestGardenDarwinLoadCommandPreservesPath(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the launchd hint targets a POSIX shell")
	}
	shell, err := exec.LookPath("sh")
	if err != nil {
		t.Skip("POSIX shell unavailable")
	}
	for _, plistPath := range []string{
		"/Users/operator/Library/LaunchAgents/com.fleet.stale-work-garden.plist",
		"/Users/`id`/Library/LaunchAgents/com.fleet.stale-work-garden.plist",
		"/Users/O'Neil & $HOME $(printf expanded) `printf backtick` \\garden/agent.plist",
	} {
		t.Run(plistPath, func(t *testing.T) {
			hint := gardenDarwinLoadCommand(plistPath)
			// A real shell parses the guidance, but both commands are shadowed by
			// shell functions. No launchctl executable or service is touched.
			script := "id() { printf '501\\n'; }\nlaunchctl() { printf '%s\\000' \"$@\"; }\n" + hint
			ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, shell, "-c", script)
			cmd.Env = []string{"PATH="}
			got, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("parse hint %q: %v; output=%q", hint, err, got)
			}
			want := "bootstrap\x00gui/501\x00" + plistPath + "\x00"
			if string(got) != want {
				t.Fatalf("load arguments = %q, want %q", got, want)
			}
		})
	}
}
