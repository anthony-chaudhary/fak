//go:build windows

package windowgate

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestOwnedProcessReapsHousekeepingDescendants(t *testing.T) {
	ps, err := exec.LookPath("powershell.exe")
	if err != nil {
		t.Skipf("powershell.exe not on PATH: %v", err)
	}

	tests := []struct {
		name         string
		rootTail     string
		trigger      func(*cleanupRecorder, context.CancelFunc)
		wantExit     int
		wantWaitErr  bool
		requireAlive bool
	}{
		{name: "success", rootTail: "exit 0", trigger: func(*cleanupRecorder, context.CancelFunc) {}, wantExit: 0},
		{name: "exit failure", rootTail: "exit 7", trigger: func(*cleanupRecorder, context.CancelFunc) {}, wantExit: 7},
		{name: "timeout", rootTail: "Start-Sleep -Seconds 120", trigger: func(*cleanupRecorder, context.CancelFunc) {}, wantExit: -1, wantWaitErr: true, requireAlive: true},
		{name: "parent cancellation", rootTail: "Start-Sleep -Seconds 120", trigger: func(_ *cleanupRecorder, cancel context.CancelFunc) { cancel() }, wantExit: -1, wantWaitErr: true, requireAlive: true},
		{name: "assertion cleanup", rootTail: "Start-Sleep -Seconds 120", trigger: func(owner *cleanupRecorder, _ context.CancelFunc) { owner.Run() }, wantExit: -1, requireAlive: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			cancel := func() {}
			if tc.name == "timeout" {
				var timeoutCancel context.CancelFunc
				ctx, timeoutCancel = context.WithTimeout(ctx, 5*time.Second)
				cancel = timeoutCancel
			} else if tc.name == "parent cancellation" {
				var parentCancel context.CancelFunc
				ctx, parentCancel = context.WithCancel(ctx)
				cancel = parentCancel
			}
			defer cancel()

			token := "FAK8371_" + strings.ReplaceAll(tc.name, " ", "_")
			childScript := fmt.Sprintf("Start-Sleep -Seconds 120 # %s", token)
			script := `$c = Start-Process -FilePath powershell.exe ` +
				fmt.Sprintf(`-ArgumentList '-NoProfile','-NonInteractive','-Command',%q `, childScript) +
				`-PassThru; Write-Output $c.Id; [Console]::Out.Flush(); ` + tc.rootTail
			cmd := exec.CommandContext(ctx, ps, "-NoProfile", "-NonInteractive", "-Command", script)
			stdout, err := cmd.StdoutPipe()
			if err != nil {
				t.Fatalf("StdoutPipe: %v", err)
			}

			owner := &cleanupRecorder{}
			process, err := StartOwnedProcess(owner, cmd)
			if err != nil {
				t.Fatalf("StartOwnedProcess: %v", err)
			}
			defer owner.Run()
			childPID := scanPID(t, stdout)
			if tc.requireAlive && (!processAlive(process.PID()) || !processAlive(childPID)) {
				t.Fatalf("tree was not alive before %s teardown: root=%d alive=%v child=%d alive=%v",
					tc.name, process.PID(), processAlive(process.PID()), childPID, processAlive(childPID))
			}

			tc.trigger(owner, cancel)
			waitErr := process.Wait()
			if tc.wantExit == 0 && waitErr != nil {
				t.Fatalf("Wait: %v", waitErr)
			}
			if tc.wantExit > 0 {
				var exitErr *exec.ExitError
				if !errors.As(waitErr, &exitErr) || exitErr.ExitCode() != tc.wantExit {
					t.Fatalf("Wait error = %T %v, want exit %d", waitErr, waitErr, tc.wantExit)
				}
			}
			if tc.wantWaitErr && waitErr == nil {
				t.Fatal("Wait returned nil after forced teardown")
			}
			if !waitFor(func() bool { return !processAlive(childPID) }, 10*time.Second) {
				t.Fatalf("housekeeping child %d survived %s teardown", childPID, tc.name)
			}
			if cmd.ProcessState == nil || !cmd.ProcessState.Exited() {
				t.Fatalf("root was not joined after %s teardown: state=%v", tc.name, cmd.ProcessState)
			}
		})
	}
}

func scanPID(t *testing.T, stdout io.Reader) int {
	t.Helper()
	result := make(chan int, 1)
	go func() {
		scanner := bufio.NewScanner(stdout)
		if scanner.Scan() {
			pid, _ := strconv.Atoi(strings.TrimSpace(scanner.Text()))
			result <- pid
			return
		}
		result <- 0
	}()
	select {
	case pid := <-result:
		if pid <= 0 {
			t.Fatal("owner did not report a housekeeping child PID")
		}
		return pid
	case <-time.After(15 * time.Second):
		t.Fatal("timed out waiting for housekeeping child PID")
		return 0
	}
}
