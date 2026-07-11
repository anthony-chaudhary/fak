package main

import (
	"errors"
	"os/exec"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// runToExit runs a command that exits with the given code and returns the
// completed run error + process state — real values from a real child, so the
// classifier is exercised against genuine *exec.ExitError / *os.ProcessState
// rather than a hand-forged one.
func runToExit(t *testing.T, code int) (error, *exec.ExitError) {
	t.Helper()
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("cmd", "/c", "exit", itoa(code))
	} else {
		cmd = exec.Command("sh", "-c", "exit "+itoa(code))
	}
	err := cmd.Run()
	if code == 0 {
		return err, nil
	}
	var ee *exec.ExitError
	if !errors.As(err, &ee) {
		t.Fatalf("exit %d did not yield an *exec.ExitError: %v", code, err)
	}
	return err, ee
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	if neg {
		b = append([]byte{'-'}, b...)
	}
	return string(b)
}

func TestGuardClassifyChildCrash(t *testing.T) {
	// Clean exit: not a crash, nothing to journal.
	cleanErr, _ := runToExit(t, 0)
	if _, _, isCrash := guardClassifyChildCrash(cleanErr, nil); isCrash {
		t.Fatal("clean exit classified as a crash")
	}

	// A nil run error (never launched / already handled) is not a crash.
	if _, _, isCrash := guardClassifyChildCrash(nil, nil); isCrash {
		t.Fatal("nil runErr classified as a crash")
	}

	// A spawn failure (not an *exec.ExitError) is reported by the caller's own path,
	// not journaled as a crash.
	if _, _, isCrash := guardClassifyChildCrash(errors.New("exec: not found"), nil); isCrash {
		t.Fatal("non-ExitError run failure classified as a crash")
	}

	// Exit 137 (128+SIGKILL) is the OOM convention.
	oomErr, oomState := runToExit(t, 137)
	class, code, isCrash := guardClassifyChildCrash(oomErr, oomState.ProcessState)
	if !isCrash || class != journal.CrashOOM || code != 137 {
		t.Fatalf("exit 137 -> class=%q code=%d crash=%v, want OOM/137/true", class, code, isCrash)
	}

	if runtime.GOOS == "windows" {
		forcedErr, forcedState := runToExit(t, -1)
		class, code, isCrash := guardClassifyChildCrash(forcedErr, forcedState.ProcessState)
		if !isCrash || class != journal.CrashSignal || code != -1 {
			t.Fatalf("Windows forced termination -> class=%q code=%d crash=%v, want SIGNAL_CRASH/-1/true", class, code, isCrash)
		}
	}

	// A plain non-zero exit (e.g. a Go panic the runtime turned into a code) is a
	// NONZERO_EXIT, distinct from a signal.
	panicErr, panicState := runToExit(t, 2)
	class, code, isCrash = guardClassifyChildCrash(panicErr, panicState.ProcessState)
	if !isCrash || class != journal.CrashNonzeroExit || code != 2 {
		t.Fatalf("exit 2 -> class=%q code=%d crash=%v, want NONZERO_EXIT/2/true", class, code, isCrash)
	}
}
