package devcmd

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/testgit"
)

func init() {
	if len(os.Args) < 2 {
		return
	}
	switch os.Args[1] {
	case "--companion-live-gate":
		os.Exit(companionLiveGateChild(os.Args[2:]))
	case "--companion-live-output-fixture":
		fmt.Fprint(os.Stdout, "fixture stdout\n")
		fmt.Fprint(os.Stderr, "fixture stderr\n")
		os.Exit(7)
	}
}

func companionLiveGateChild(argv []string) int {
	// The parent already ran TestMain and retains its checkout identity tripwire.
	// Copy its isolated global config into the parent-owned child TMPDIR, clearing
	// repository locator variables again without a pre-dispatch Git subprocess.
	cleanup, err := testgit.IsolateGit()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	defer cleanup() // This returning helper cleans up before init calls os.Exit.
	return RunCompanionGate(os.Stdout, os.Stderr, argv)
}

const companionLiveOutputLimit = 1 << 20

type companionLiveOutput struct {
	mu       sync.Mutex
	data     []byte
	overflow bool
}

func (w *companionLiveOutput) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	n := len(p)
	room := companionLiveOutputLimit - len(w.data)
	if n > room {
		w.overflow = true
	}
	w.data = append(w.data, p[:min(room, n)]...)
	return n, nil
}

// An ordinary exit can be wrapped by the owned runner's errors.Join. Never let
// errors.As(ExitError) hide a simultaneous cleanup or inherited-pipe error.
func companionLiveExitCode(err error) (int, bool) {
	if err == nil {
		return 0, true
	}
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		children := joined.Unwrap()
		if len(children) == 1 {
			return companionLiveExitCode(children[0])
		}
		return 0, false
	}
	if exited, ok := err.(*exec.ExitError); ok {
		return exited.ExitCode(), true
	}
	return 0, false
}

func runCompanionLiveNative(t *testing.T, argv ...string) (string, string, int) {
	t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	if end, ok := t.Deadline(); ok && end.Add(-2*time.Second).Before(deadline) {
		deadline = end.Add(-2 * time.Second)
	}
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()
	select {
	case companionProbeSlot <- struct{}{}:
		defer func() { <-companionProbeSlot }()
	case <-ctx.Done():
		t.Fatal("live Gate deadline exhausted waiting for subprocess slot")
	}
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.CommandContext(ctx, os.Args[0], argv...)
	cmd.Dir = cwd
	cmd.Env = append(companionProbeEnv(t.TempDir()), "GOFLAGS="+os.Getenv("GOFLAGS")+" -p=1")
	cmd.Stdin = os.Stdin
	cmd.WaitDelay = time.Second
	var stdout, stderr companionLiveOutput
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err = runOwnedCompanionProbe(cmd)
	if ctx.Err() != nil {
		t.Fatalf("live Gate subprocess deadline: %v", ctx.Err())
	}
	code, ordinary := companionLiveExitCode(err)
	if !ordinary {
		t.Fatalf("live Gate subprocess ownership/execution failure: %v", err)
	}
	// Wait joined both copy loops before these buffers are inspected.
	if stdout.overflow || stderr.overflow {
		t.Fatalf("live Gate output exceeded %d bytes per stream", companionLiveOutputLimit)
	}
	return string(stdout.data), string(stderr.data), code
}

func runCompanionLiveGate(t *testing.T, stdout, stderr io.Writer, argv []string) int {
	t.Helper()
	out, diagnostic, code := runCompanionLiveNative(t, append([]string{"--companion-live-gate"}, argv...)...)
	if _, err := io.WriteString(stdout, out); err != nil {
		t.Fatal(err)
	}
	if _, err := io.WriteString(stderr, diagnostic); err != nil {
		t.Fatal(err)
	}
	return code
}

// fak-test:runtime fast est=1s lane=default
func TestCompanionGateLiveCapture(t *testing.T) {
	if _, ordinary := companionLiveExitCode(errors.Join(&exec.ExitError{}, exec.ErrWaitDelay)); ordinary {
		t.Fatal("joined cleanup failure was hidden by child exit")
	}
	if _, ordinary := companionLiveExitCode(exec.ErrWaitDelay); ordinary {
		t.Fatal("pipe timeout was treated as ordinary exit")
	}
	var output companionLiveOutput
	input := strings.Repeat("x", companionLiveOutputLimit+1)
	if n, err := output.Write([]byte(input)); err != nil || n != len(input) || len(output.data) != companionLiveOutputLimit || !output.overflow {
		t.Fatalf("capture did not retain overflow evidence: n=%d err=%v bytes=%d overflow=%v", n, err, len(output.data), output.overflow)
	}
	out, diagnostic, code := runCompanionLiveNative(t, "--companion-live-output-fixture")
	if out != "fixture stdout\n" || diagnostic != "fixture stderr\n" || code != 7 {
		t.Fatalf("native output/exit changed: stdout=%q stderr=%q code=%d", out, diagnostic, code)
	}
}
