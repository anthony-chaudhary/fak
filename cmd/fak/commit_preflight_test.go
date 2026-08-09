package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// withCommitPreflightFn swaps the commitPreflightFn seam for the duration of a test, so
// runCommitPreflight is exercised with a canned report and never touches git or a repo.
func withCommitPreflightFn(t *testing.T, fn func(context.Context, string, []string) (safecommit.PathPreflightReport, error)) {
	t.Helper()
	prev := commitPreflightFn
	commitPreflightFn = fn
	t.Cleanup(func() { commitPreflightFn = prev })
}

func TestRunCommitPreflight_noPathsIsUsageError(t *testing.T) {
	withCommitPreflightFn(t, func(context.Context, string, []string) (safecommit.PathPreflightReport, error) {
		t.Fatal("classifier must not run without a path")
		return safecommit.PathPreflightReport{}, nil
	})
	var out, errb bytes.Buffer
	if code := runCommitPreflight(&out, &errb, nil); code != 2 {
		t.Fatalf("want exit 2 for no paths, got %d (stderr=%q)", code, errb.String())
	}
}

// TestRunCommitPreflight_allTrackedExit0 is the DoD clean witness at the CLI seam.
func TestRunCommitPreflight_allTrackedExit0(t *testing.T) {
	var gotPaths []string
	withCommitPreflightFn(t, func(_ context.Context, _ string, paths []string) (safecommit.PathPreflightReport, error) {
		gotPaths = paths
		return safecommit.PathPreflightReport{
			OK: true,
			Classes: []safecommit.PathClass{
				{Path: "a.go", State: safecommit.PathTracked},
				{Path: "b.go", State: safecommit.PathTracked},
			},
		}, nil
	})
	var out, errb bytes.Buffer
	code := runCommitPreflight(&out, &errb, []string{"--path", "a.go", "--", "b.go"})
	if code != 0 {
		t.Fatalf("want exit 0 for all-tracked, got %d (stderr=%q)", code, errb.String())
	}
	if len(gotPaths) != 2 || gotPaths[0] != "a.go" || gotPaths[1] != "b.go" {
		t.Fatalf("shim should pass --path AND positional pathspecs, got %v", gotPaths)
	}
	if !strings.Contains(out.String(), "OK") {
		t.Fatalf("clean human output should say OK, got %q", out.String())
	}
}

// TestRunCommitPreflight_untrackedIsRefusedAndNamesFix is the DoD refusal witness: an untracked
// path exits with the refused-on-the-merits code and the human output names the path and the
// `git add` fix.
func TestRunCommitPreflight_untrackedIsRefusedAndNamesFix(t *testing.T) {
	withCommitPreflightFn(t, func(context.Context, string, []string) (safecommit.PathPreflightReport, error) {
		return safecommit.PathPreflightReport{
			OK:        false,
			Reason:    safecommit.ReasonPathUntracked,
			Untracked: []string{"cmd/fak/new.go"},
			Classes: []safecommit.PathClass{
				{Path: "cmd/fak/new.go", State: safecommit.PathUntracked, Fix: "git add -- cmd/fak/new.go   (present in the worktree but not staged)"},
			},
		}, nil
	})
	var out, errb bytes.Buffer
	code := runCommitPreflight(&out, &errb, []string{"--path", "cmd/fak/new.go"})
	if code != safecommit.ExitRefused {
		t.Fatalf("want exit %d for a refusal on the merits (never the retryable contention 3), got %d",
			safecommit.ExitRefused, code)
	}
	got := out.String()
	if !strings.Contains(got, "cmd/fak/new.go") || !strings.Contains(got, "git add") {
		t.Fatalf("refusal output must name the path and the git add fix, got %q", got)
	}
	if !strings.Contains(got, safecommit.ReasonPathUntracked) {
		t.Fatalf("refusal output must name the reason, got %q", got)
	}
}

// TestRunCommitPreflight_jsonShape confirms --json emits the machine contract a worker branches
// on: ok=false, the named reason, and the per-path fix.
func TestRunCommitPreflight_jsonShape(t *testing.T) {
	withCommitPreflightFn(t, func(context.Context, string, []string) (safecommit.PathPreflightReport, error) {
		return safecommit.PathPreflightReport{
			OK:        false,
			Reason:    safecommit.ReasonPathUnmatched,
			Unmatched: []string{"stale/plan.go"},
			Detail:    "1 pathspec(s) match nothing git knows: stale/plan.go",
			Classes: []safecommit.PathClass{
				{Path: "stale/plan.go", State: safecommit.PathUnmatched, Fix: `no file matches "stale/plan.go" — check for a typo, a renamed/moved path, or a stale plan entry`},
			},
		}, nil
	})
	var out, errb bytes.Buffer
	code := runCommitPreflight(&out, &errb, []string{"--json", "--path", "stale/plan.go"})
	if code != safecommit.ExitRefused {
		t.Fatalf("want exit %d, got %d", safecommit.ExitRefused, code)
	}
	var rep safecommit.PathPreflightReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("stdout is not the JSON report: %v (out=%q)", err, out.String())
	}
	if rep.OK || rep.Reason != safecommit.ReasonPathUnmatched {
		t.Fatalf("json must carry ok=false and the reason, got %+v", rep)
	}
	if len(rep.Classes) != 1 || rep.Classes[0].Fix == "" {
		t.Fatalf("json must carry the per-path fix, got %+v", rep.Classes)
	}
}

func TestRunCommitPreflight_infraErrorExit1(t *testing.T) {
	withCommitPreflightFn(t, func(context.Context, string, []string) (safecommit.PathPreflightReport, error) {
		return safecommit.PathPreflightReport{}, errors.New("git not executable")
	})
	var out, errb bytes.Buffer
	if code := runCommitPreflight(&out, &errb, []string{"--path", "a.go"}); code != 1 {
		t.Fatalf("want exit 1 for an infra error, got %d", code)
	}
	if !strings.Contains(errb.String(), "git not executable") {
		t.Fatalf("stderr should surface the infra error, got %q", errb.String())
	}
}
