package main

import (
	"bytes"
	"context"
	"errors"
	"runtime"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func setDispatchProcessProbeStubs(t *testing.T, pooled, oneShot func(string, time.Duration, bool) ([]byte, error)) {
	t.Helper()
	oldPooled := dispatchRunHostProbeFunc
	oldOneShot := dispatchRunHostProbeOneShotFunc
	dispatchRunHostProbeFunc = pooled
	dispatchRunHostProbeOneShotFunc = oneShot
	t.Cleanup(func() {
		dispatchRunHostProbeFunc = oldPooled
		dispatchRunHostProbeOneShotFunc = oldOneShot
	})
}

func TestDispatchScanProcessesWindowsRetriesEmptyWarmFrameOneShot(t *testing.T) {
	pooledCalls, oneShotCalls := 0, 0
	setDispatchProcessProbeStubs(t,
		func(string, time.Duration, bool) ([]byte, error) { pooledCalls++; return nil, nil },
		func(string, time.Duration, bool) ([]byte, error) {
			oneShotCalls++
			return []byte(`[{"pid":7,"name":"worker","threads":2,"handles":3,"ws_mb":4}]`), nil
		},
	)
	got, err := dispatchScanProcessesWindows()
	if err != nil || pooledCalls != 1 || oneShotCalls != 1 || len(got) != 1 || got[0].PID != 7 {
		t.Fatalf("pooled=%d one-shot=%d got=%+v err=%v", pooledCalls, oneShotCalls, got, err)
	}
}

func TestDispatchScanProcessesWindowsReturnsTypedErrorAfterTwoEmptyFrames(t *testing.T) {
	pooledCalls, oneShotCalls := 0, 0
	setDispatchProcessProbeStubs(t,
		func(string, time.Duration, bool) ([]byte, error) { pooledCalls++; return []byte{}, nil },
		func(string, time.Duration, bool) ([]byte, error) { oneShotCalls++; return []byte{}, nil },
	)
	_, err := dispatchScanProcessesWindows()
	if err == nil || pooledCalls != 1 || oneShotCalls != 1 {
		t.Fatalf("pooled=%d one-shot=%d err=%v", pooledCalls, oneShotCalls, err)
	}
}

func TestDispatchScanProcessesWindowsReturnsOneShotRetryError(t *testing.T) {
	want := errors.New("probe retry failed")
	setDispatchProcessProbeStubs(t,
		func(string, time.Duration, bool) ([]byte, error) { return nil, nil },
		func(string, time.Duration, bool) ([]byte, error) { return nil, want },
	)
	_, err := dispatchScanProcessesWindows()
	if !errors.Is(err, want) {
		t.Fatalf("err=%v", err)
	}
}

func TestDispatchHostProbePowerShellKeepsUsablePipesOnWindows(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows process handles only")
	}
	cmd := windowgate.CommandContext(context.Background(), "powershell", "-NoProfile", "-NonInteractive", "-Command", `Write-Output '{"ok":true}'`)
	if cmd.SysProcAttr == nil || cmd.SysProcAttr.CreationFlags&windowgate.CreateNoWindow == 0 {
		t.Fatalf("host probe flags=%#v, want CREATE_NO_WINDOW", cmd.SysProcAttr)
	}
	if cmd.SysProcAttr.CreationFlags&windowgate.DetachedProcess != 0 {
		t.Fatalf("host probe flags=%#x include DETACHED_PROCESS, which drops redirected PowerShell output", cmd.SysProcAttr.CreationFlags)
	}
	out, err := cmd.Output()
	if err != nil || !bytes.Contains(out, []byte(`{"ok":true}`)) {
		t.Fatalf("out=%q err=%v", out, err)
	}
}
