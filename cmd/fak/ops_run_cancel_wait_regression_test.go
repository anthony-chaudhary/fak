package main

import (
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"runtime"
	"testing"
	"time"
)

type opsCancelWaitWriter func([]byte) (int, error)

func (w opsCancelWaitWriter) Write(p []byte) (int, error) {
	return w(p)
}

// fak-test:runtime medium est=5s lane=default
func TestOpsCancelWaitRegression(t *testing.T) {
	const childEnv = "FAK_OPS_CANCEL_WAIT_REGRESSION_CHILD"
	if mode := os.Getenv(childEnv); mode != "" {
		_, _ = os.Stdout.WriteString("ready\n")
		if mode == "hold" {
			// Bound an orphan even if the parent fails.
			time.Sleep(10 * time.Second)
		}
		os.Exit(17)
	}

	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	argv := []string{exe, "-test.run=^TestOpsCancelWaitRegression$"}
	childEnvFor := func(mode string) []string {
		return append(os.Environ(), childEnv+"="+mode)
	}

	// The probe must be safe while the real os/exec waiter publishes its
	// ProcessState. No test-side reads of ProcessState occur before Wait.
	t.Run("probe_during_wait_and_after_exit", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
		cmd.Env = childEnvFor("exit")
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		done := make(chan error, 1)
		go func() { done <- cmd.Wait() }()
		for {
			state := probeChildProcessState(cmd)
			if state != "alive" && state != "exited" {
				t.Errorf("started child state = %q", state)
			}
			select {
			case err := <-done:
				if ctx.Err() != nil {
					t.Fatalf("helper exceeded its deadline: %v", ctx.Err())
				}
				var exitErr *exec.ExitError
				if !errors.As(err, &exitErr) || exitErr.ExitCode() != 17 {
					t.Fatalf("helper exit = %v, want exit 17", err)
				}
				if got := probeChildProcessState(cmd); got != "exited" {
					t.Fatalf("reaped child state = %q, want exited", got)
				}
				return
			default:
				runtime.Gosched()
			}
		}
	})

	for _, engine := range []struct {
		name string
		run  func(context.Context, io.Writer, io.Writer, []string, []string, []byte) (int, bool, bool, []opsRunLifecycleRecord)
	}{
		{"opencode", executeOpsRun},
		{"pi", opsPiExecute},
	} {
		for _, mode := range []string{"cancel", "deadline", "kill_error"} {
			t.Run(engine.name+"/"+mode, func(t *testing.T) {
				// These tests replace a package seam, so must not run in parallel.
				original := opsRunCancelProcess
				defer func() { opsRunCancelProcess = original }()
				injected := errors.New("cancel-wait regression: kill failed")
				captured := make(chan *exec.Cmd, 1)
				opsRunCancelProcess = func(cmd *exec.Cmd, cancel func() error) error {
					captured <- cmd
					err := original(cmd, cancel)
					// Keep Cancel active while Wait reaps the killed child.
					// Sleep introduces no synchronization with ProcessState.
					time.Sleep(25 * time.Millisecond)
					if mode == "kill_error" {
						return injected
					}
					return err
				}

				ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
				defer cancel()
				out := opsCancelWaitWriter(func(p []byte) (int, error) {
					if mode != "deadline" {
						cancel()
					}
					return len(p), nil
				})
				code, complete, failed, lifecycle := engine.run(
					ctx, out, io.Discard, argv, childEnvFor("hold"), nil,
				)
				if code == 0 || complete || failed {
					t.Fatalf("result = (%d, %v, %v), want interrupted non-event child",
						code, complete, failed)
				}
				if len(lifecycle) != 1 {
					t.Fatalf("lifecycle = %+v, want one cancellation record", lifecycle)
				}
				rec := lifecycle[0]
				wantReason := "cancelled"
				if mode == "deadline" {
					wantReason = "timeout"
					if !errors.Is(ctx.Err(), context.DeadlineExceeded) {
						t.Fatalf("context error = %v, want deadline", ctx.Err())
					}
				}
				if rec.Reason != wantReason || rec.TerminationReason != wantReason {
					t.Fatalf("termination reason = %+v, want %q", rec, wantReason)
				}
				if rec.ChildState != "alive" || rec.State != "alive" || rec.Signal != "SIGKILL" {
					t.Fatalf("cancellation attribution = %+v", rec)
				}
				wantError := ""
				if mode == "kill_error" {
					wantError = injected.Error()
				}
				if rec.Error != wantError || rec.OSError != wantError {
					t.Fatalf("kill error = %+v, want %q", rec, wantError)
				}
				select {
				case cmd := <-captured:
					// Both Run and its Cancel callback have returned.
					if cmd.ProcessState == nil || rec.ExitCode == nil {
						t.Fatalf("missing final process state or exit code: %+v", rec)
					}
					if *rec.ExitCode != cmd.ProcessState.ExitCode() {
						t.Fatalf("receipt exit = %d, process exit = %d",
							*rec.ExitCode, cmd.ProcessState.ExitCode())
					}
				default:
					t.Fatal("cancellation callback was not invoked")
				}
				if rec.Duration == "" || rec.ElapsedMS < 0 || rec.DurationMS != rec.ElapsedMS {
					t.Fatalf("invalid final duration: %+v", rec)
				}
			})
		}
	}
}
