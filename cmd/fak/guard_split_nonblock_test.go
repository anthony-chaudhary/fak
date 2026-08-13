package main

import (
	"bytes"
	"os/exec"
	"runtime"
	"testing"
	"time"
)

// TestOpenGuardInfoPaneNonBlocking dogfoods the startup fix on the real host: the multiplexer
// client spawn (on Windows `wt.exe` is an AppX exec alias with a ~200ms cold start) must NOT
// block the agent launch. We point the execCommand seam at a ~2s command and assert
// openGuardInfoPane returns in well under that — proof the wt/tmux spawn is fire-and-reap
// (cmd.Start + a background cmd.Wait), not cmd.Run() sitting on the critical path between the
// gateway going healthy and the agent starting.
func TestOpenGuardInfoPaneNonBlocking(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("uses the win32 `ping` sleeper as the slow-spawn fixture; the non-blocking property itself is host-agnostic")
	}
	savedGOOS, savedLook, savedExec := guardSplitGOOS, guardSplitLookPath, execCommand
	t.Cleanup(func() { guardSplitGOOS, guardSplitLookPath, execCommand = savedGOOS, savedLook, savedExec })

	// Force the Windows Terminal (wt) plan regardless of the real host multiplexer.
	guardSplitGOOS, guardSplitLookPath = "windows", lookPathOK
	// A spawn that takes ~2s to COMPLETE. If openGuardInfoPane waited on it (the old cmd.Run()),
	// the call would take ~2s; with the shipped cmd.Start() it returns in milliseconds.
	execCommand = func(string, ...string) *exec.Cmd {
		return exec.Command("ping", "-n", "3", "127.0.0.1")
	}

	var buf bytes.Buffer
	start := time.Now()
	openGuardInfoPane(&buf, envFunc(map[string]string{"WT_SESSION": "x"}), "bottom", "http://127.0.0.1:9", 2*time.Second)
	elapsed := time.Since(start)

	if elapsed > 1*time.Second {
		t.Fatalf("openGuardInfoPane blocked for %v while the spawn takes ~2s — it must NOT wait on the multiplexer client (non-blocking startup regressed)", elapsed)
	}
	// Success is visible as the pane itself; compact startup must not narrate it into
	// the agent surface. Only an actionable spawn failure belongs on stderr.
	if got := buf.String(); got != "" {
		t.Fatalf("successful split polluted compact startup: %q", got)
	}
}
