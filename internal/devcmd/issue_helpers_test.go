package devcmd

import (
	"bytes"
	"flag"
	"io"
	"strings"
	"testing"
)

// TestParseFlagsRejectArgs pins the (exitCode, done) contract every flags-only
// dev front door reads. The bool is DONE-shaped, not ok-shaped: true means the
// caller returns exitCode immediately. It went untested and inverted long enough
// to break `issue-contract-repair` outright (valid flags exited 0 having done
// nothing; a positional argument sailed past the check into the command body), so
// the direction is pinned here rather than left to the reader of two call sites.
func TestParseFlagsRejectArgs(t *testing.T) {
	newFS := func() (*flag.FlagSet, *int) {
		fs := flag.NewFlagSet("probe", flag.ContinueOnError)
		n := fs.Int("n", 0, "a number")
		return fs, n
	}

	t.Run("valid flags proceed", func(t *testing.T) {
		fs, n := newFS()
		var errOut bytes.Buffer
		code, done := parseFlagsRejectArgs(fs, []string{"--n", "7"}, &errOut)
		if done {
			t.Fatal("done = true on a clean parse: the command body would never run")
		}
		if code != 0 {
			t.Errorf("code = %d, want 0", code)
		}
		if *n != 7 {
			t.Errorf("--n = %d, want 7 (flags must actually bind)", *n)
		}
	})

	t.Run("positional argument is refused", func(t *testing.T) {
		fs, _ := newFS()
		var errOut bytes.Buffer
		code, done := parseFlagsRejectArgs(fs, []string{"--n", "7", "stray"}, &errOut)
		if !done {
			t.Fatal("done = false on a positional: it would fall through into the command")
		}
		if code != 2 {
			t.Errorf("code = %d, want 2", code)
		}
		if !strings.Contains(errOut.String(), "unexpected positional arguments: stray") {
			t.Errorf("stderr must name the offending word, got %q", errOut.String())
		}
	})

	t.Run("unknown flag stops the command", func(t *testing.T) {
		fs, _ := newFS()
		fs.SetOutput(io.Discard)
		code, done := parseFlagsRejectArgs(fs, []string{"--nope"}, io.Discard)
		if !done || code != 2 {
			t.Errorf("(code, done) = (%d, %v), want (2, true)", code, done)
		}
	})

	t.Run("help request stops the command", func(t *testing.T) {
		fs, _ := newFS()
		fs.SetOutput(io.Discard)
		code, done := parseFlagsRejectArgs(fs, []string{"-h"}, io.Discard)
		if !done || code != 2 {
			t.Errorf("(code, done) = (%d, %v), want (2, true)", code, done)
		}
	})
}
