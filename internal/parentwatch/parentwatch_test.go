package parentwatch

import (
	"context"
	"io"
	"os"
	"os/exec"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

// helperProcessEnv gates the re-exec'd test binary into a plain child process.
// The value selects the child's lifetime: "block" waits for stdin EOF, "exit"
// returns at once. This is a portable stand-in for sh/sleep (absent on
// Windows) and lets the fixture terminate the child without process signals,
// which are not portable across POSIX and Windows.
const helperProcessEnv = "PARENTWATCH_HELPER_PROCESS"

// TestHelperProcess is not a real test: it is the body of the portable child
// fixture. It runs only when explicitly re-exec'd with helperProcessEnv set.
func TestHelperProcess(t *testing.T) {
	switch os.Getenv(helperProcessEnv) {
	case "block":
		_, _ = io.Copy(io.Discard, os.Stdin)
		os.Exit(0)
	case "exit":
		os.Exit(0)
	}
}

// child is a live helper process plus the write end of its lifetime pipe.
type child struct {
	cmd   *exec.Cmd
	stdin io.WriteCloser
}

// startBlockingChild starts a child that stays alive until terminate closes its
// stdin pipe. Portable replacement for `sleep 30`.
func startBlockingChild(t *testing.T) *child {
	t.Helper()
	stdin, childStdin, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	cmd := exec.Command(os.Args[0], "-test.run=TestHelperProcess")
	cmd.Env = append(os.Environ(), helperProcessEnv+"=block")
	cmd.Stdin = childStdin
	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		_ = childStdin.Close()
		t.Fatalf("start helper child: %v", err)
	}
	_ = childStdin.Close()
	return &child{cmd: cmd, stdin: stdin}
}

// terminate ends the child by closing its lifetime pipe and reaps it, so its
// pid is guaranteed dead without platform-specific signal handling.
func (c *child) terminate(t *testing.T) {
	t.Helper()
	_ = c.stdin.Close()
	_ = c.cmd.Wait()
}

// runToExit starts a child that exits immediately, then reaps it so its pid is
// guaranteed dead. Portable replacement for `sh -c "exit 0"`.
func runToExit(t *testing.T) int {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=TestHelperProcess")
	cmd.Env = append(os.Environ(), helperProcessEnv+"=exit")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start helper child: %v", err)
	}
	pid := cmd.Process.Pid
	if err := cmd.Wait(); err != nil {
		t.Fatalf("reap helper child: %v", err)
	}
	return pid
}

func TestParentAlive(t *testing.T) {
	if !ParentAlive(os.Getpid()) {
		t.Fatal("ParentAlive(self) = false, want true")
	}
	if ParentAlive(0) || ParentAlive(-1) {
		t.Fatal("ParentAlive(non-positive pid) = true, want false")
	}

	pid := runToExit(t)
	if ParentAlive(pid) {
		t.Fatalf("ParentAlive(reaped pid %d) = true, want false", pid)
	}
}

func TestWatchLiveParent(t *testing.T) {
	c := startBlockingChild(t)
	pid := c.cmd.Process.Pid

	ctx, stop := Watch(context.Background(), pid)
	defer stop()

	select {
	case <-ctx.Done():
		t.Fatal("ctx canceled while parent alive")
	case <-time.After(300 * time.Millisecond):
	}

	c.terminate(t)

	select {
	case <-ctx.Done():
	case <-time.After(3 * time.Second):
		t.Fatal("ctx not canceled after parent death")
	}
}

func TestWatchAlreadyDeadParent(t *testing.T) {
	pid := runToExit(t)

	ctx, stop := Watch(context.Background(), pid)
	defer stop()

	select {
	case <-ctx.Done():
	case <-time.After(500 * time.Millisecond):
		t.Fatal("ctx not canceled for already-dead parent")
	}
}

// TestWatchReusedPIDCancels is the PID-reuse witness: the watched PID is live,
// but the recorded creation time is stale (as if the original parent exited and
// the number was recycled). Watch must cancel on the identity mismatch even
// though the PID-only probe still reports the number alive. Where the platform
// exposes no creation time (non-Windows), the guard cannot arm, so the test
// asserts the documented PID-only fallback instead of a false positive.
func TestWatchReusedPIDCancels(t *testing.T) {
	c := startBlockingChild(t)
	defer c.terminate(t)
	pid := c.cmd.Process.Pid

	live, ok := processalive.StartTime(pid)
	if !ok {
		// No creation time on this platform: the reuse guard is unavailable and
		// Watch falls back to the PID-only probe. The live PID must NOT cancel.
		ctx, stop := Watch(context.Background(), pid)
		defer stop()
		select {
		case <-ctx.Done():
			t.Fatal("ctx canceled for a live PID despite no StartTime guard")
		case <-time.After(300 * time.Millisecond):
		}
		t.Skip("processalive.StartTime unavailable; reuse guard not armable")
	}

	stale := live.Add(-time.Hour)
	ctx, stop := WatchIdent(context.Background(), pid, stale, true)
	defer stop()

	select {
	case <-ctx.Done():
	case <-time.After(3 * time.Second):
		t.Fatal("ctx not canceled for a reused PID (stale StartTime, live PID)")
	}
}

// TestWatchMatchingStartTimeLive confirms the guard does not misfire when the
// recorded creation time matches the live process.
func TestWatchMatchingStartTimeLive(t *testing.T) {
	c := startBlockingChild(t)
	pid := c.cmd.Process.Pid

	live, ok := processalive.StartTime(pid)
	if !ok {
		t.Skip("processalive.StartTime unavailable")
	}

	ctx, stop := WatchIdent(context.Background(), pid, live, true)
	defer stop()

	select {
	case <-ctx.Done():
		t.Fatal("ctx canceled while parent alive with matching StartTime")
	case <-time.After(300 * time.Millisecond):
	}

	c.terminate(t)

	select {
	case <-ctx.Done():
	case <-time.After(3 * time.Second):
		t.Fatal("ctx not canceled after parent death with matching StartTime")
	}
}

func TestWatchNoParent(t *testing.T) {
	parent := context.Background()
	ctx, stop := Watch(parent, 1)
	if ctx != parent {
		t.Fatal("Watch(pid<=1) returned a different context")
	}
	if ctx.Err() != nil {
		t.Fatal("Watch(pid<=1) context already canceled")
	}
	stop()
	stop()

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	ctx2, stop2 := Watch(canceled, 0)
	if ctx2 != canceled {
		t.Fatal("Watch(parentPid<=1) did not return parent")
	}
	if ctx2.Err() == nil {
		t.Fatal("returned parent ctx not canceled")
	}
	stop2()
	stop2()
}

func TestStopIdempotent(t *testing.T) {
	c := startBlockingChild(t)
	defer c.terminate(t)

	_, stop := Watch(context.Background(), c.cmd.Process.Pid)
	stop()
	stop()
}
