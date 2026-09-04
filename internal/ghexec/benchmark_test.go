package ghexec

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"testing"
	"time"
)

var benchSinkCmd *exec.Cmd

// BenchmarkCommand measures the baseline Command construction throughput
// and allocations for typical gh CLI invocations.
func BenchmarkCommand(b *testing.B) {
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cmd := Command(ctx, "issue", "list", "--json", "number,state,title")
		if cmd == nil || len(cmd.Args) == 0 {
			b.Fatal("unexpected nil or empty command")
		}
		benchSinkCmd = cmd
	}
}

// BenchmarkCommandTimeout measures CommandTimeout construction across nil
// and explicit parent context configurations, including deadline derivation
// and cancel function lifecycles.
func BenchmarkCommandTimeout(b *testing.B) {
	b.Run("NilParent", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cmd, cancel := CommandTimeout(nil, DefaultTimeout, "pr", "view", "42")
			if cmd == nil {
				b.Fatal("unexpected nil command")
			}
			cancel()
			benchSinkCmd = cmd
		}
	})

	b.Run("WithParent", func(b *testing.B) {
		ctx := context.Background()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cmd, cancel := CommandTimeout(ctx, DefaultTimeout, "pr", "view", "42")
			if cmd == nil {
				b.Fatal("unexpected nil command")
			}
			cancel()
			benchSinkCmd = cmd
		}
	})
}

// BenchmarkCommandHardwareScrubbing measures argument scrubbing throughput
// during Command construction for clean vs leak-sensitive PR/issue bodies.
func BenchmarkCommandHardwareScrubbing(b *testing.B) {
	cleanBody := "Bug report: unexpected EOF during token streaming on client connection."
	leakBody := "Run comparison on da" + "33 against dgx" + "1 cluster with sxm" + "4 configuration."

	b.Run("CleanBody", func(b *testing.B) {
		ctx := context.Background()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cmd := Command(ctx, "issue", "comment", "101", "--body", cleanBody)
			if cmd == nil {
				b.Fatal("unexpected nil command")
			}
			benchSinkCmd = cmd
		}
	})

	b.Run("ScrubbedBody", func(b *testing.B) {
		ctx := context.Background()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cmd := Command(ctx, "issue", "comment", "101", "--body", leakBody)
			if cmd == nil {
				b.Fatal("unexpected nil command")
			}
			benchSinkCmd = cmd
		}
	})
}

// BenchmarkCommandParallel measures concurrent command construction across
// multiple goroutines simulating concurrent agent dispatchers.
func BenchmarkCommandParallel(b *testing.B) {
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			cmd, cancel := CommandTimeout(ctx, DefaultTimeout, "issue", "list", "--limit", "50")
			if cmd == nil {
				b.Fatal("unexpected nil command")
			}
			cancel()
		}
	})
}

// BenchmarkCommandFailFastDeadline measures the fail-fast execution path
// when a deadlined command is invoked after its deadline has already expired.
func BenchmarkCommandFailFastDeadline(b *testing.B) {
	exe, err := os.Executable()
	if err != nil {
		b.Fatalf("os.Executable: %v", err)
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cmd, cancel := CommandTimeout(ctx, -time.Nanosecond, "issue", "list")
		cmd.Path = exe
		cmd.Err = nil
		runErr := cmd.Run()
		cancel()
		if !errors.Is(runErr, context.DeadlineExceeded) {
			b.Fatalf("expected context.DeadlineExceeded, got %v", runErr)
		}
	}
}

// BenchmarkCommandArgScaling measures command construction latency and memory
// allocations across varying argument counts.
func BenchmarkCommandArgScaling(b *testing.B) {
	for _, count := range []int{2, 10, 50} {
		args := make([]string, count)
		args[0] = "issue"
		args[1] = "edit"
		for j := 2; j < count; j++ {
			args[j] = fmt.Sprintf("--add-label=label-%d", j)
		}
		b.Run(fmt.Sprintf("Args_%d", count), func(b *testing.B) {
			ctx := context.Background()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				cmd := Command(ctx, args...)
				if len(cmd.Args) == 0 {
					b.Fatal("unexpected empty args")
				}
				benchSinkCmd = cmd
			}
		})
	}
}
