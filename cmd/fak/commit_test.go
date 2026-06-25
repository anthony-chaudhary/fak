package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// withCommitFn swaps the commitFn seam for the duration of a test.
func withCommitFn(t *testing.T, fn func(context.Context, safecommit.Options) (safecommit.Result, error)) {
	t.Helper()
	prev := commitFn
	commitFn = fn
	t.Cleanup(func() { commitFn = prev })
}

func TestRunCommit_noPathsIsUsageError(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"-m", "msg"})
	if code != 2 {
		t.Fatalf("want exit 2 for no paths, got %d (stderr=%q)", code, errb.String())
	}
}

func TestRunCommit_noMessageIsUsageError(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go"})
	if code != 2 {
		t.Fatalf("want exit 2 for no message, got %d (stderr=%q)", code, errb.String())
	}
}

func TestRunCommit_dashMAndDashFAreExclusive(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go", "-m", "x", "-F", "f.txt"})
	if code != 2 {
		t.Fatalf("want exit 2 for -m+-F, got %d", code)
	}
	if !strings.Contains(errb.String(), "mutually exclusive") {
		t.Fatalf("stderr should explain the conflict, got %q", errb.String())
	}
}

func TestRunCommit_positionalPathsAfterDashDash(t *testing.T) {
	var gotOpts safecommit.Options
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		gotOpts = o
		return safecommit.Result{Committed: true, Verified: true, SHA: "abc", Paths: o.Paths}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"-m", "msg", "--", "internal/x.go", "internal/y.go"})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (stderr=%q)", code, errb.String())
	}
	if len(gotOpts.Paths) != 2 {
		t.Fatalf("positional paths after -- should reach Options, got %v", gotOpts.Paths)
	}
	if !gotOpts.SignOff {
		t.Fatalf("sign-off should default on")
	}
}

func TestRunCommit_jsonShapeAndRaceExitCode(t *testing.T) {
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{
			Committed:  true,
			Verified:   false,
			SHA:        "deadbeefcafe",
			Paths:      o.Paths,
			Reason:     safecommit.ReasonPathspecRace,
			RacedExtra: []string{"internal/peer/swept.go"},
			HeadBefore: "0000111122223333",
		}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--json", "--path", "a.go", "-m", "msg"})
	// PATHSPEC_RACE: the commit ran but is bad -> exit 1 (halt).
	if code != 1 {
		t.Fatalf("race should exit 1, got %d", code)
	}
	var res safecommit.Result
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("--json must emit a valid Result: %v\noutput=%q", err, out.String())
	}
	if res.Reason != safecommit.ReasonPathspecRace || len(res.RacedExtra) != 1 {
		t.Fatalf("json result lost the race evidence: %+v", res)
	}
}

func TestRunCommit_offTrunkExit3(t *testing.T) {
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{Reason: safecommit.ReasonOffTrunk, Detail: "on feature/x, expected main", Paths: o.Paths}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go", "-m", "msg"})
	if code != 3 {
		t.Fatalf("a pre-commit refusal should exit 3, got %d", code)
	}
}

func TestRunCommit_infraErrorExit1(t *testing.T) {
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{}, errTest
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go", "-m", "msg"})
	if code != 1 {
		t.Fatalf("an infra error should exit 1, got %d", code)
	}
}

func TestRunCommit_messageFromStdin(t *testing.T) {
	var gotMsg string
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		gotMsg = o.Message
		return safecommit.Result{Committed: true, Verified: true, Paths: o.Paths}, nil
	})
	prev := stdin
	stdin = func() io.Reader { return strings.NewReader("from stdin — em-dash ok\n") }
	t.Cleanup(func() { stdin = prev })

	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go", "-F", "-"})
	if code != 0 {
		t.Fatalf("want 0, got %d (stderr=%q)", code, errb.String())
	}
	if !strings.Contains(gotMsg, "from stdin") {
		t.Fatalf("message should come from stdin, got %q", gotMsg)
	}
}

var errTest = errTestErr{}

type errTestErr struct{}

func (errTestErr) Error() string { return "test infra failure" }
