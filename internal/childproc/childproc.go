package childproc

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"sync"
	"time"
)

// DefaultMaxOutputBytes is the default buffer ceiling for stdout and stderr (10 MiB).
const DefaultMaxOutputBytes int64 = 10 * 1024 * 1024

// DefaultWaitDelay is the default grace duration to wait for I/O completion after
// sending process cancellation before forcibly tearing down remaining resources.
const DefaultWaitDelay time.Duration = 2 * time.Second

// ExitCodeUnknown indicates that a process failed to start or terminated abnormally
// without providing an exit status code.
const ExitCodeUnknown int = -1

// Result contains the final execution state, streams, and metrics of a finished or supervised process.
type Result struct {
	// ExitCode is the numerical exit status returned by the process.
	// Invariant: ExitCode is 0 if and only if the process exited cleanly with success and did not time out.
	// Guard: fail-closed exit code propagation (-1 on launch failure, signal death, timeout, or abort).
	ExitCode int

	// Stdout contains the captured standard output bytes from the process.
	Stdout []byte

	// Stderr contains the captured standard error bytes from the process.
	Stderr []byte

	// Duration records the total wall-clock elapsed execution time.
	Duration time.Duration

	// TimedOut indicates whether the process was terminated due to exceeding its execution deadline.
	TimedOut bool
}

// Success returns true if the process completed with an exit code of 0 and did not time out.
func (r *Result) Success() bool {
	return r != nil && r.ExitCode == 0 && !r.TimedOut
}

// StdoutString returns the captured stdout as a UTF-8 string.
func (r *Result) StdoutString() string {
	if r == nil {
		return ""
	}
	return string(r.Stdout)
}

// StderrString returns the captured stderr as a UTF-8 string.
func (r *Result) StderrString() string {
	if r == nil {
		return ""
	}
	return string(r.Stderr)
}

// Err returns an error if the process timed out, failed to launch, or finished with a non-zero exit code.
// Invariant: Err() returns nil if and only if Success() returns true.
// Guard: fail-closed exit code propagation: non-zero exit or abort yields a descriptive error.
func (r *Result) Err() error {
	if r == nil {
		return errors.New("childproc: nil result")
	}
	if r.TimedOut {
		return context.DeadlineExceeded
	}
	if r.ExitCode != 0 {
		return fmt.Errorf("childproc: process exited with code %d", r.ExitCode)
	}
	return nil
}

// Command specifies the configuration and execution limits for a supervised child process.
type Command struct {
	// Path or name of the binary executable.
	Path string

	// Args holds command-line arguments to pass to the binary (excluding binary name).
	Args []string

	// Dir specifies the working directory for execution. If empty, the current working directory is used.
	Dir string

	// Env provides environment variables in KEY=VALUE format. If nil, host process environment is inherited.
	Env []string

	// Stdin supplies input bytes to the child process standard input.
	Stdin io.Reader

	// Timeout bounds the maximum wall-clock duration for process execution.
	// If 0, execution deadline is governed solely by the passed context.
	Timeout time.Duration

	// MaxOutputBytes limits the maximum number of bytes captured in Stdout and Stderr.
	// If 0, DefaultMaxOutputBytes is used. If negative, capture is unbounded.
	MaxOutputBytes int64

	// WaitDelay is the grace period before force-closing pipes after process termination.
	// If 0, DefaultWaitDelay is used.
	WaitDelay time.Duration

	// FailOnNonZeroExit causes Run to return an error when the process exits with a non-zero code.
	FailOnNonZeroExit bool
}

// NewCommand creates a new Command instance initialized with executable path and arguments.
func NewCommand(cmd string, args ...string) *Command {
	return &Command{
		Path:           cmd,
		Args:           append([]string(nil), args...),
		MaxOutputBytes: DefaultMaxOutputBytes,
		WaitDelay:      DefaultWaitDelay,
	}
}

// Run executes a command with arguments under context supervision and returns the resulting execution metrics.
// Invariant: Duration is non-negative and represents monotonic elapsed execution time.
// Guard: fail-closed exit code propagation: on any start error, timeout, or abort, ExitCode is guaranteed to be non-zero (-1).
func Run(ctx context.Context, cmd string, args ...string) (*Result, error) {
	return NewCommand(cmd, args...).Run(ctx)
}

// Run executes the Command under context supervision, capturing output streams and handling timeouts.
// Invariant: Duration is non-negative and represents monotonic elapsed execution time.
// Guard: fail-closed exit code propagation: on any start error, timeout, or abort, ExitCode is guaranteed to be non-zero (-1).
func (c *Command) Run(ctx context.Context) (*Result, error) {
	if c == nil {
		return nil, errors.New("childproc: nil command")
	}
	if c.Path == "" {
		return nil, errors.New("childproc: empty command path")
	}
	if ctx == nil {
		ctx = context.Background()
	}

	// Fail-closed check: if context is already expired before spawning, abort immediately.
	if err := ctx.Err(); err != nil {
		timedOut := errors.Is(err, context.DeadlineExceeded)
		return &Result{
			ExitCode: ExitCodeUnknown,
			Duration: 0,
			TimedOut: timedOut,
		}, err
	}

	runCtx := ctx
	var cancel context.CancelFunc
	if c.Timeout > 0 {
		runCtx, cancel = context.WithTimeout(ctx, c.Timeout)
		defer cancel()
	}

	maxBytes := c.MaxOutputBytes
	if maxBytes == 0 {
		maxBytes = DefaultMaxOutputBytes
	}

	stdoutBuf := newBoundedBuffer(maxBytes)
	stderrBuf := newBoundedBuffer(maxBytes)

	cmd := exec.CommandContext(runCtx, c.Path, c.Args...)
	cmd.Dir = c.Dir
	cmd.Env = c.Env
	cmd.Stdin = c.Stdin
	cmd.Stdout = stdoutBuf
	cmd.Stderr = stderrBuf

	waitDelay := c.WaitDelay
	if waitDelay <= 0 {
		waitDelay = DefaultWaitDelay
	}
	cmd.WaitDelay = waitDelay

	ConfigureProcessGroup(cmd)

	start := time.Now()
	err := cmd.Start()
	if err != nil {
		duration := time.Since(start)
		res := &Result{
			ExitCode: ExitCodeUnknown,
			Stdout:   stdoutBuf.Bytes(),
			Stderr:   stderrBuf.Bytes(),
			Duration: duration,
			TimedOut: false,
		}
		return res, err
	}

	waitErr := cmd.Wait()
	duration := time.Since(start)

	timedOut := errors.Is(runCtx.Err(), context.DeadlineExceeded)
	ctxErr := runCtx.Err()

	exitCode := extractExitCode(waitErr, timedOut, ctxErr)

	res := &Result{
		ExitCode: exitCode,
		Stdout:   stdoutBuf.Bytes(),
		Stderr:   stderrBuf.Bytes(),
		Duration: duration,
		TimedOut: timedOut,
	}

	if timedOut {
		return res, context.DeadlineExceeded
	}
	if errors.Is(ctxErr, context.Canceled) {
		return res, context.Canceled
	}
	if waitErr != nil {
		var exitErr *exec.ExitError
		if errors.As(waitErr, &exitErr) {
			if c.FailOnNonZeroExit {
				return res, waitErr
			}
			return res, nil
		}
		return res, waitErr
	}

	return res, nil
}

// extractExitCode extracts the integer exit code from process execution error or state.
// Invariant: extractExitCode preserves the exact exit code reported by the operating system.
// Guard: fail-closed exit code propagation: unresolvable exit status defaults to -1.
func extractExitCode(waitErr error, timedOut bool, ctxErr error) int {
	if timedOut || ctxErr != nil {
		return ExitCodeUnknown
	}
	if waitErr == nil {
		return 0
	}
	var exitErr *exec.ExitError
	if errors.As(waitErr, &exitErr) {
		return exitErr.ExitCode()
	}
	return ExitCodeUnknown
}

// boundedBuffer provides thread-safe bounded memory storage for captured process streams.
// Invariant: Len() never exceeds limit when limit is positive.
// Guard: fail-closed exit code propagation unaffected by buffer truncation.
type boundedBuffer struct {
	mu        sync.Mutex
	buf       bytes.Buffer
	limit     int64 // <= 0 means unbounded
	truncated bool
}

func newBoundedBuffer(limit int64) *boundedBuffer {
	return &boundedBuffer{
		limit: limit,
	}
}

func (b *boundedBuffer) Write(p []byte) (n int, err error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.limit <= 0 {
		return b.buf.Write(p)
	}

	room := b.limit - int64(b.buf.Len())
	if room <= 0 {
		b.truncated = true
		return len(p), nil
	}

	if int64(len(p)) > room {
		b.truncated = true
		_, _ = b.buf.Write(p[:room])
		return len(p), nil
	}

	return b.buf.Write(p)
}

func (b *boundedBuffer) Bytes() []byte {
	b.mu.Lock()
	defer b.mu.Unlock()

	out := make([]byte, b.buf.Len())
	copy(out, b.buf.Bytes())
	return out
}

func (b *boundedBuffer) Truncated() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.truncated
}
