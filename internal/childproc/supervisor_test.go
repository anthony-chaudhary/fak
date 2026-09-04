package childproc

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"testing"
	"time"
)

func TestTimeoutHandling(t *testing.T) {
	ctx := context.Background()
	cmd := newTestHelperCommand("sleep")
	cmd.Timeout = 150 * time.Millisecond
	cmd.WaitDelay = 500 * time.Millisecond

	start := time.Now()
	res, err := cmd.Run(ctx)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want context.DeadlineExceeded", err)
	}
	if res == nil {
		t.Fatal("expected non-nil Result")
	}
	if !res.TimedOut {
		t.Errorf("TimedOut = false, want true")
	}
	// Guard: fail-closed exit code propagation: timed-out command reports non-zero exit code (-1).
	if res.ExitCode != ExitCodeUnknown {
		t.Errorf("ExitCode = %d, want %d", res.ExitCode, ExitCodeUnknown)
	}
	if res.Success() {
		t.Errorf("Success() = true, want false")
	}
	if elapsed >= 10*time.Second {
		t.Errorf("elapsed = %v, expected termination well before 10s", elapsed)
	}
}

func TestContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cmd := newTestHelperCommand("sleep")
	cmd.WaitDelay = 500 * time.Millisecond

	go func() {
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	res, err := cmd.Run(ctx)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected cancellation error, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if res == nil {
		t.Fatal("expected non-nil Result")
	}
	if res.TimedOut {
		t.Errorf("TimedOut = true, want false")
	}
	// Guard: fail-closed exit code propagation
	if res.ExitCode != ExitCodeUnknown {
		t.Errorf("ExitCode = %d, want %d", res.ExitCode, ExitCodeUnknown)
	}
	if res.Success() {
		t.Errorf("Success() = true, want false")
	}
	if elapsed >= 10*time.Second {
		t.Errorf("elapsed = %v, expected termination well before 10s", elapsed)
	}
}

func TestPreCanceledAndExpiredContext(t *testing.T) {
	// Pre-canceled context
	ctxCanceled, cancel := context.WithCancel(context.Background())
	cancel()

	cmd := newTestHelperCommand("echo")
	res, err := cmd.Run(ctxCanceled)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
	if res == nil || res.ExitCode != ExitCodeUnknown {
		t.Errorf("res.ExitCode = %v, want %d", res, ExitCodeUnknown)
	}

	// Pre-expired context
	ctxExpired, cancelExpired := context.WithDeadline(context.Background(), time.Now().Add(-1*time.Second))
	defer cancelExpired()

	resExp, errExp := cmd.Run(ctxExpired)
	if !errors.Is(errExp, context.DeadlineExceeded) {
		t.Errorf("errExp = %v, want context.DeadlineExceeded", errExp)
	}
	if resExp == nil || !resExp.TimedOut || resExp.ExitCode != ExitCodeUnknown {
		t.Errorf("resExp = %+v, want TimedOut=true, ExitCode=%d", resExp, ExitCodeUnknown)
	}
}

func TestExitCodeExtraction(t *testing.T) {
	ctx := context.Background()

	cases := []struct {
		name     string
		code     string
		wantCode int
	}{
		{name: "exit 0", code: "0", wantCode: 0},
		{name: "exit 1", code: "1", wantCode: 1},
		{name: "exit 42", code: "42", wantCode: 42},
		{name: "exit 127", code: "127", wantCode: 127},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cmd := newTestHelperCommand("exit", helperCodeEnv+"="+tc.code)
			res, err := cmd.Run(ctx)
			if err != nil {
				t.Fatalf("unexpected execution error: %v", err)
			}
			if res == nil {
				t.Fatal("expected non-nil Result")
			}
			if res.ExitCode != tc.wantCode {
				t.Errorf("ExitCode = %d, want %d", res.ExitCode, tc.wantCode)
			}
			if tc.wantCode == 0 {
				if !res.Success() {
					t.Errorf("Success() = false for exit 0")
				}
				if res.Err() != nil {
					t.Errorf("res.Err() = %v, want nil", res.Err())
				}
			} else {
				if res.Success() {
					t.Errorf("Success() = true for exit %d", tc.wantCode)
				}
				if res.Err() == nil {
					t.Errorf("res.Err() = nil, want error for exit %d", tc.wantCode)
				}
			}
		})
	}
}

func TestFailOnNonZeroExit(t *testing.T) {
	ctx := context.Background()
	cmd := newTestHelperCommand("exit", helperCodeEnv+"=42")
	cmd.FailOnNonZeroExit = true

	res, err := cmd.Run(ctx)
	if err == nil {
		t.Fatal("expected error with FailOnNonZeroExit=true, got nil")
	}
	if res == nil {
		t.Fatal("expected non-nil Result")
	}
	if res.ExitCode != 42 {
		t.Errorf("ExitCode = %d, want 42", res.ExitCode)
	}
}

func TestExtractExitCodeContract(t *testing.T) {
	// Invariant & Guard verification for extractExitCode
	if code := extractExitCode(nil, false, nil); code != 0 {
		t.Errorf("clean exit extractExitCode = %d, want 0", code)
	}
	if code := extractExitCode(nil, true, nil); code != ExitCodeUnknown {
		t.Errorf("timed out extractExitCode = %d, want %d", code, ExitCodeUnknown)
	}
	if code := extractExitCode(nil, false, context.Canceled); code != ExitCodeUnknown {
		t.Errorf("canceled extractExitCode = %d, want %d", code, ExitCodeUnknown)
	}
	if code := extractExitCode(errors.New("non-exit error"), false, nil); code != ExitCodeUnknown {
		t.Errorf("generic error extractExitCode = %d, want %d", code, ExitCodeUnknown)
	}
}

func TestProcessTreeTerminationHelper(t *testing.T) {
	// Test safe handling of non-positive pids
	if err := KillTree(0); err != nil {
		t.Errorf("KillTree(0) error = %v, want nil", err)
	}
	if err := KillTree(-1); err != nil {
		t.Errorf("KillTree(-1) error = %v, want nil", err)
	}

	// Test safe handling of nil command
	if err := TerminateProcessTree(nil); err != nil {
		t.Errorf("TerminateProcessTree(nil) error = %v, want nil", err)
	}
	var unstarted exec.Cmd
	if err := TerminateProcessTree(&unstarted); err != nil {
		t.Errorf("TerminateProcessTree(unstarted) error = %v, want nil", err)
	}

	// Test real tree spawn and termination
	cmd := exec.CommandContext(context.Background(), os.Args[0], "-test.run=^TestHelperProcess$")
	cmd.Env = append(os.Environ(), helperEnv+"=grandchild")
	ConfigureProcessGroup(cmd)
	if err := cmd.Start(); err != nil {
		t.Fatalf("cmd.Start() failed: %v", err)
	}

	pid := cmd.Process.Pid
	if err := KillTree(pid); err != nil {
		t.Errorf("KillTree(pid) = %v", err)
	}
	_ = cmd.Wait()
}
