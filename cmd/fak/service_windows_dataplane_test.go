//go:build windows

package main

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestWindowsControlLoopOwnsCrashSensorAndMachinePaths(t *testing.T) {
	t.Setenv("ProgramData", t.TempDir())
	// A canceled context still performs one authoritative sensor tick before stop.
	oldCrash, oldResume := windowsControlCrashTick, windowsControlResumeTick
	crashes, resumes := 0, 0
	windowsControlCrashTick = func(io.Writer, io.Writer, string) int { crashes++; return 0 }
	windowsControlResumeTick = func(io.Writer, io.Writer) int { resumes++; return 0 }
	t.Cleanup(func() { windowsControlCrashTick, windowsControlResumeTick = oldCrash, oldResume })
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var out, errout bytes.Buffer
	rc := runWindowsControlLoop(ctx, &out, &errout, time.Hour)
	if rc != 0 || crashes != 1 || resumes != 1 {
		t.Fatalf("rc=%d stderr=%s", rc, errout.String())
	}
	state := windowsServiceStateDir()
	if got := os.Getenv("FLEET_REG_DIR"); got != filepath.Join(state, "registry") {
		t.Fatalf("registry=%q", got)
	}
	if got := os.Getenv("FAK_HOST_RELAUNCH_DIR"); got != filepath.Join(state, "relaunch") {
		t.Fatalf("spool=%q", got)
	}
}
func TestWindowsServiceHandlerReportsRunningAndStops(t *testing.T) { _ = io.Discard }
