//go:build windows

package systembaseline

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

const (
	windowsJobHelperRole = "FAK_SYSTEMBASELINE_JOB_HELPER"
	windowsJobHelperFan  = "FAK_SYSTEMBASELINE_JOB_FANOUT"
)

func TestWindowsJobObjectExactDescendantsAndCleanup(t *testing.T) {
	if role := os.Getenv(windowsJobHelperRole); role != "" {
		runWindowsJobHelper(role)
		return
	}

	const fanout = 12
	tests := []struct {
		name          string
		role          string
		timeout       time.Duration
		wantExit      int
		wantProcesses uint64
		wantKilled    bool
	}{
		{name: "success", role: "root-success", wantExit: 0, wantProcesses: 1 + 2*fanout},
		{name: "failure", role: "root-failure", wantExit: 7, wantProcesses: 1 + 2*fanout},
		{name: "timeout", role: "root-timeout", timeout: 2 * time.Second, wantExit: -1, wantProcesses: 2, wantKilled: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			cancel := func() {}
			if tc.timeout > 0 {
				ctx, cancel = context.WithTimeout(ctx, tc.timeout)
			}
			defer cancel()

			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestWindowsJobObjectExactDescendantsAndCleanup$")
			cmd.Env = append(os.Environ(), windowsJobHelperRole+"="+tc.role, windowsJobHelperFan+"="+strconv.Itoa(fanout))
			attributor := NewCommandAttributor()
			if !attributor.Configure(cmd) {
				receipt := attributor.FinishAttribution()
				t.Fatalf("Windows Job Object unavailable on integration host: %+v", receipt.WindowsJobObject)
			}
			if err := cmd.Start(); err != nil {
				attributor.LaunchFailed(err)
				t.Fatalf("start helper: %v", err)
			}
			if err := attributor.Started(cmd.Process.Pid); err != nil {
				_ = cmd.Process.Kill()
				_ = cmd.Wait()
				t.Fatalf("assign/readback/resume helper: %v", err)
			}
			waitErr := cmd.Wait()
			if tc.wantExit == 0 && waitErr != nil {
				t.Fatalf("helper Wait: %v", waitErr)
			}
			if tc.wantExit > 0 {
				var exitErr *exec.ExitError
				if !errors.As(waitErr, &exitErr) || exitErr.ExitCode() != tc.wantExit {
					t.Fatalf("helper Wait error=%T %v, want exit %d", waitErr, waitErr, tc.wantExit)
				}
			}
			if tc.wantExit < 0 && waitErr == nil {
				t.Fatal("timed helper unexpectedly succeeded")
			}

			receipt := attributor.FinishAttribution()
			job := receipt.WindowsJobObject
			if job == nil {
				t.Fatal("missing Windows Job Object receipt")
			}
			if job.State != WindowsJobStateMeasured || !job.Membership.AtomicPlacement || job.Membership.RootPID != cmd.Process.Pid || job.Membership.RootStartID == 0 {
				t.Fatalf("unverified job membership: %+v", job)
			}
			if got := job.Processes.Values["total_processes"]; got != tc.wantProcesses {
				t.Fatalf("lifetime descendants=%d, want %d (100%% root/child/grandchild attribution); receipt=%+v", got, tc.wantProcesses, job)
			}
			if job.Cleanup.KilledRemaining != tc.wantKilled || !job.Cleanup.Empty || !job.Cleanup.Closed || job.Cleanup.Reason != "" {
				t.Fatalf("cleanup=%+v, want killed=%v empty+closed", job.Cleanup, tc.wantKilled)
			}
			if err := job.validate(); err != nil {
				t.Fatalf("receipt validation: %v; receipt=%+v", err, job)
			}
		})
	}
}

func TestWindowsJobAssignmentDenialResumesSampledFallback(t *testing.T) {
	originalAssign := assignWindowsProcessToJob
	assignWindowsProcessToJob = func(syscall.Handle, syscall.Handle) error {
		return errors.New("injected nested-job permission denial")
	}
	t.Cleanup(func() { assignWindowsProcessToJob = originalAssign })

	cmd := exec.Command("cmd.exe", "/d", "/c", "exit", "0")
	attributor := NewCommandAttributor()
	if !attributor.Configure(cmd) {
		t.Fatal("job setup unavailable before injected assignment denial")
	}
	if err := cmd.Start(); err != nil {
		attributor.LaunchFailed(err)
		t.Fatal(err)
	}
	if err := attributor.Started(cmd.Process.Pid); err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		t.Fatalf("assignment denial should resume sampled fallback, got fatal error: %v", err)
	}
	if err := cmd.Wait(); err != nil {
		t.Fatalf("sampled fallback child was not resumed: %v", err)
	}
	receipt := attributor.FinishAttribution().WindowsJobObject
	if receipt == nil || receipt.State != WindowsJobStateUnavailable || !strings.Contains(receipt.Reason, "using sampled PID/PPID coverage") || !receipt.Cleanup.Empty || !receipt.Cleanup.Closed {
		t.Fatalf("assignment denial was not explicit sampled fallback: %+v", receipt)
	}
}

func runWindowsJobHelper(role string) {
	child := func(next string) *exec.Cmd {
		cmd := exec.Command(os.Args[0], "-test.run=^TestWindowsJobObjectExactDescendantsAndCleanup$")
		cmd.Env = append(os.Environ(), windowsJobHelperRole+"="+next)
		return cmd
	}
	switch role {
	case "root-success", "root-failure":
		fanout, _ := strconv.Atoi(os.Getenv(windowsJobHelperFan))
		children := make([]*exec.Cmd, 0, fanout)
		for i := 0; i < fanout; i++ {
			cmd := child("child")
			if err := cmd.Start(); err != nil {
				os.Exit(20)
			}
			children = append(children, cmd)
		}
		for _, cmd := range children {
			if err := cmd.Wait(); err != nil {
				os.Exit(21)
			}
		}
		if role == "root-failure" {
			os.Exit(7)
		}
		os.Exit(0)
	case "child":
		if err := child("grandchild").Run(); err != nil {
			os.Exit(22)
		}
		os.Exit(0)
	case "grandchild":
		os.Exit(0)
	case "root-timeout":
		if err := child("sleeper").Start(); err != nil {
			os.Exit(23)
		}
		time.Sleep(2 * time.Minute)
		os.Exit(0)
	case "sleeper":
		time.Sleep(2 * time.Minute)
		os.Exit(0)
	default:
		os.Exit(24)
	}
}
