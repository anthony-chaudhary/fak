package toolproc

import (
	"bytes"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestProcessSupervisor(t *testing.T) {
	t.Run("ExitTombstone", func(t *testing.T) {
		sup := NewProcessSupervisor()
		pid := 101
		cmd := "worker --task=fetch"

		handle := sup.RegisterProcess(pid, cmd)
		if handle == nil || handle.PID != pid || handle.State != ProcessRunning {
			t.Fatalf("unexpected handle: %+v", handle)
		}
		if sup.ActiveCount() != 1 {
			t.Fatalf("expected 1 active process, got %d", sup.ActiveCount())
		}

		exitCode := 1
		stderr := "fatal error: network unreachable"
		stdout := "attempt 1 failed"
		elapsed := 120 * time.Millisecond

		sup.RecordExit(pid, exitCode, stdout, stderr, elapsed)

		if sup.ActiveCount() != 0 {
			t.Fatalf("expected 0 active processes after exit, got %d", sup.ActiveCount())
		}
		if sup.TombstoneCount() != 1 {
			t.Fatalf("expected 1 tombstone, got %d", sup.TombstoneCount())
		}

		tb, ok := sup.GetTombstone(pid)
		if !ok || tb == nil {
			t.Fatalf("expected tombstone for PID %d to exist", pid)
		}
		if tb.PID != pid {
			t.Errorf("expected PID %d, got %d", pid, tb.PID)
		}
		if tb.State != ProcessFailed {
			t.Errorf("expected state %s, got %s", ProcessFailed, tb.State)
		}
		if tb.ExitCode != 1 {
			t.Errorf("expected exit code 1, got %d", tb.ExitCode)
		}
		if tb.StderrTail != stderr {
			t.Errorf("expected stderr tail %q, got %q", stderr, tb.StderrTail)
		}
		if tb.StdoutTail != stdout {
			t.Errorf("expected stdout tail %q, got %q", stdout, tb.StdoutTail)
		}
		if tb.Elapsed != elapsed {
			t.Errorf("expected elapsed %v, got %v", elapsed, tb.Elapsed)
		}
		if tb.ConsecutivePolls != 0 {
			t.Errorf("expected 0 initial consecutive polls, got %d", tb.ConsecutivePolls)
		}
		if tb.ConsecutiveWrites != 0 {
			t.Errorf("expected 0 initial consecutive writes, got %d", tb.ConsecutiveWrites)
		}
	})

	t.Run("PollTombstonedProcess", func(t *testing.T) {
		sup := NewProcessSupervisor()
		pid := 201

		sup.RegisterProcess(pid, "task")
		sup.RecordExit(pid, 0, "completed successfully", "", 45*time.Millisecond)

		status, err := sup.PollProcess(pid)
		if err != nil {
			t.Fatalf("unexpected error polling tombstoned process: %v", err)
		}
		if status.PID != pid {
			t.Errorf("expected PID %d, got %d", pid, status.PID)
		}
		if status.State != ProcessExited {
			t.Errorf("expected state %s, got %s", ProcessExited, status.State)
		}
		if status.ExitCode != 0 {
			t.Errorf("expected exit code 0, got %d", status.ExitCode)
		}
		if status.StdoutTail != "completed successfully" {
			t.Errorf("expected stdout %q, got %q", "completed successfully", status.StdoutTail)
		}
		if status.StderrTail != "" {
			t.Errorf("expected empty stderr, got %q", status.StderrTail)
		}
		if status.Elapsed != 45*time.Millisecond {
			t.Errorf("expected elapsed %v, got %v", 45*time.Millisecond, status.Elapsed)
		}
	})

	t.Run("WriteStdinDeadProcess", func(t *testing.T) {
		sup := NewProcessSupervisor()
		pid := 301
		stderr := "pipe closed unexpectedly"

		sup.RegisterProcess(pid, "repl")
		sup.RecordExit(pid, 2, "", stderr, 80*time.Millisecond)

		err := sup.WriteStdin(pid, []byte("help\n"))
		if err == nil {
			t.Fatal("expected error writing stdin to dead process, got nil")
		}

		var termErr *ErrProcessTerminated
		if !errors.As(err, &termErr) {
			t.Fatalf("expected *ErrProcessTerminated, got %T: %v", err, err)
		}
		if termErr.PID != pid {
			t.Errorf("expected PID %d, got %d", pid, termErr.PID)
		}
		if termErr.ExitCode != 2 {
			t.Errorf("expected exit code 2, got %d", termErr.ExitCode)
		}
		if termErr.StderrTail != stderr {
			t.Errorf("expected stderr %q, got %q", stderr, termErr.StderrTail)
		}

		expectedMsg := "PROCESS_TERMINATED: PID 301 exited with code 2; stderr: pipe closed unexpectedly"
		if err.Error() != expectedMsg {
			t.Errorf("expected error string %q, got %q", expectedMsg, err.Error())
		}
	})

	t.Run("LivelockSuppression", func(t *testing.T) {
		t.Run("Polling", func(t *testing.T) {
			sup := NewProcessSupervisor(WithPollLivelockThreshold(4))
			pid := 401
			sup.RegisterProcess(pid, "loop")
			sup.RecordExit(pid, 1, "", "crash", 10*time.Millisecond)

			// Poll attempts 1 through 4 should return terminal status without error
			for i := 1; i <= 4; i++ {
				status, err := sup.PollProcess(pid)
				if err != nil {
					t.Fatalf("poll attempt %d failed unexpectedly: %v", i, err)
				}
				if status.State != ProcessFailed {
					t.Errorf("poll attempt %d: expected state %s, got %s", i, ProcessFailed, status.State)
				}
			}

			// Attempt 5 exceeds threshold 4 -> trips livelock suppression
			status, err := sup.PollProcess(pid)
			if err == nil {
				t.Fatal("expected livelock suppression error on attempt 5, got nil")
			}
			var liveErr *ErrLivelockSuppressed
			if !errors.As(err, &liveErr) {
				t.Fatalf("expected *ErrLivelockSuppressed, got %T: %v", err, err)
			}
			if liveErr.PID != pid {
				t.Errorf("expected PID %d, got %d", pid, liveErr.PID)
			}
			expectedMsg := "LIVELOCK_SUPPRESSED: repeated polling on terminated process PID 401"
			if err.Error() != expectedMsg {
				t.Errorf("expected error %q, got %q", expectedMsg, err.Error())
			}
			if status.State != ProcessFailed {
				t.Errorf("expected status state to still report %s, got %s", ProcessFailed, status.State)
			}
		})

		t.Run("WriteCircuitBreaker", func(t *testing.T) {
			sup := NewProcessSupervisor(WithWriteLivelockThreshold(4))
			pid := 402
			sup.RegisterProcess(pid, "dead_writer")
			sup.RecordExit(pid, 1, "", "oom", 15*time.Millisecond)

			// Attempts 1 through 4 should return PROCESS_TERMINATED
			for i := 1; i <= 4; i++ {
				err := sup.WriteStdin(pid, []byte("data\n"))
				if err == nil {
					t.Fatalf("write attempt %d succeeded unexpectedly", i)
				}
				var termErr *ErrProcessTerminated
				if !errors.As(err, &termErr) {
					t.Fatalf("attempt %d: expected *ErrProcessTerminated, got %T: %v", i, err, err)
				}
			}

			// Attempt 5 exceeds threshold 4 -> trips circuit breaker
			err := sup.WriteStdin(pid, []byte("data\n"))
			if err == nil {
				t.Fatal("expected circuit breaker error on attempt 5, got nil")
			}
			var circuitErr *ErrLivelockCircuitBroken
			if !errors.As(err, &circuitErr) {
				t.Fatalf("expected *ErrLivelockCircuitBroken, got %T: %v", err, err)
			}
			if circuitErr.PID != pid {
				t.Errorf("expected PID %d, got %d", pid, circuitErr.PID)
			}
			expectedMsg := "LIVELOCK_CIRCUIT_BROKEN: PID 402 is dead"
			if err.Error() != expectedMsg {
				t.Errorf("expected error %q, got %q", expectedMsg, err.Error())
			}
		})
	})

	t.Run("ActiveProcessOperations", func(t *testing.T) {
		sup := NewProcessSupervisor()
		pid := 501
		cmd := "server --port 8080"

		handle := sup.RegisterProcess(pid, cmd)
		var stdinBuf bytes.Buffer
		handle.SetStdin(&stdinBuf)

		// Polling active process returns running state
		status, err := sup.PollProcess(pid)
		if err != nil {
			t.Fatalf("unexpected poll error: %v", err)
		}
		if status.State != ProcessRunning {
			t.Errorf("expected state %s, got %s", ProcessRunning, status.State)
		}
		if status.PID != pid {
			t.Errorf("expected PID %d, got %d", pid, status.PID)
		}

		// Writing to active process writes to stdin
		data := []byte("ping\n")
		if err := sup.WriteStdin(pid, data); err != nil {
			t.Fatalf("failed to write stdin to active process: %v", err)
		}
		if stdinBuf.String() != "ping\n" {
			t.Errorf("expected stdin buffer %q, got %q", "ping\n", stdinBuf.String())
		}
	})

	t.Run("UnknownProcess", func(t *testing.T) {
		sup := NewProcessSupervisor()
		unknownPID := 9999

		_, err := sup.PollProcess(unknownPID)
		if err == nil {
			t.Fatal("expected error polling unknown PID, got nil")
		}
		var unknownErr *ErrUnknownProcess
		if !errors.As(err, &unknownErr) {
			t.Fatalf("expected *ErrUnknownProcess, got %T: %v", err, err)
		}
		if unknownErr.PID != unknownPID {
			t.Errorf("expected PID %d, got %d", unknownPID, unknownErr.PID)
		}
		if !errors.Is(err, ErrUnknownProcessSentinel) {
			t.Errorf("expected errors.Is(err, ErrUnknownProcessSentinel) to match")
		}

		err = sup.WriteStdin(unknownPID, []byte("data"))
		if err == nil {
			t.Fatal("expected error writing to unknown PID, got nil")
		}
		if !errors.As(err, &unknownErr) {
			t.Fatalf("expected *ErrUnknownProcess, got %T: %v", err, err)
		}
	})

	t.Run("TombstoneFIFOEviction", func(t *testing.T) {
		sup := NewProcessSupervisor(WithTombstoneCapacity(3))

		// Register and exit 4 processes
		for i := 1; i <= 4; i++ {
			sup.RegisterProcess(i, "cmd")
			sup.RecordExit(i, 0, "", "", 10*time.Millisecond)
		}

		if sup.TombstoneCount() != 3 {
			t.Fatalf("expected tombstone count 3, got %d", sup.TombstoneCount())
		}

		// PID 1 should have been evicted FIFO
		if _, ok := sup.GetTombstone(1); ok {
			t.Errorf("expected PID 1 to have been evicted")
		}

		// PIDs 2, 3, 4 should still exist
		for _, pid := range []int{2, 3, 4} {
			if _, ok := sup.GetTombstone(pid); !ok {
				t.Errorf("expected PID %d to be retained in tombstones", pid)
			}
		}
	})

	t.Run("TailTruncation", func(t *testing.T) {
		sup := NewProcessSupervisor(WithMaxTailBytes(64))
		pid := 601
		longStderr := strings.Repeat("x", 200) + "END_OF_STDERR"

		sup.RegisterProcess(pid, "verbose")
		sup.RecordExit(pid, 1, "", longStderr, 10*time.Millisecond)

		tb, ok := sup.GetTombstone(pid)
		if !ok {
			t.Fatal("expected tombstone")
		}
		if len(tb.StderrTail) > 64 {
			t.Errorf("expected stderr tail <= 64 bytes, got %d", len(tb.StderrTail))
		}
		if !strings.HasSuffix(tb.StderrTail, "END_OF_STDERR") {
			t.Errorf("expected stderr tail to end with %q, got %q", "END_OF_STDERR", tb.StderrTail)
		}
	})

	t.Run("TimeoutState", func(t *testing.T) {
		sup := NewProcessSupervisor()
		pid := 701

		sup.RegisterProcess(pid, "sleep 100")
		sup.RecordTimeout(pid, "", "timed out after 5s", 5*time.Second)

		status, err := sup.PollProcess(pid)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if status.State != ProcessTimedOut {
			t.Errorf("expected state %s, got %s", ProcessTimedOut, status.State)
		}
		if status.ExitCode != -1 {
			t.Errorf("expected exit code -1, got %d", status.ExitCode)
		}
	})
}
