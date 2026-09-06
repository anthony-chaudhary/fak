package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

func TestRunCommitHelpDocumentsBuildCheckTimeout(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--help"})
	if code != 2 {
		t.Fatalf("help exit = %d, want usage exit 2", code)
	}
	for _, want := range []string{
		"build-check-timeout",
		"default 4m",
		"controls prospective validation, not advisory-lock waiting or earlier build/materialization phases",
	} {
		if !strings.Contains(errb.String(), want) {
			t.Fatalf("commit help missing %q; got:\n%s", want, errb.String())
		}
	}
}

func TestRunCommitRejectsNonPositiveBuildCheckTimeout(t *testing.T) {
	for _, flagVal := range []string{"--build-check-timeout=0", "--build-check-timeout=0s", "--build-check-timeout=-1s"} {
		var out, errb bytes.Buffer
		code := runCommit(&out, &errb, []string{flagVal, "--path", "a.go", "-m", "fix: test (fak a)"})
		if code != 2 || !strings.Contains(errb.String(), "fak commit: --build-check-timeout must be greater than zero") {
			t.Fatalf("flag %q: code=%d stderr=%q, want greater than zero refusal", flagVal, code, errb.String())
		}
	}
}

func TestRunCommitRejectsMalformedOrOverflowBuildCheckTimeout(t *testing.T) {
	for _, flagVal := range []string{"--build-check-timeout=notaduration", "--build-check-timeout=99999999999999999999h"} {
		var out, errb bytes.Buffer
		code := runCommit(&out, &errb, []string{flagVal, "--path", "a.go", "-m", "fix: test (fak a)"})
		if code != 2 {
			t.Fatalf("flag %q: code=%d, want usage exit 2", flagVal, code)
		}
	}
}

func TestRunCommitForwardsBuildCheckTimeout(t *testing.T) {
	oldBuild := commitBuildCheckGateWithTimeout
	t.Cleanup(func() { commitBuildCheckGateWithTimeout = oldBuild })

	var recordedTimeout time.Duration
	commitBuildCheckGateWithTimeout = func(_ io.Writer, _ string, _ []string, timeout time.Duration) (safecommit.BuildCheckOutcome, string) {
		recordedTimeout = timeout
		return safecommit.BuildCheckPassed, ""
	}

	withCommitFn(t, func(_ context.Context, opts safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{Committed: true, Verified: true, SHA: "abc1234", Paths: opts.Paths}, nil
	})

	// Case 1: explicit positive timeout reaches prospective validation
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--path", "a.go", "-m", "fix(test): forward timeout (fak cmd)",
		"--build-check-timeout", "12m",
	})
	if code != 0 {
		t.Fatalf("runCommit exit = %d, stderr = %q", code, errb.String())
	}
	if recordedTimeout != 12*time.Minute {
		t.Fatalf("recordedTimeout = %v, want 12m", recordedTimeout)
	}

	// Case 2: omitted timeout defaults to 4m
	recordedTimeout = 0
	out.Reset()
	errb.Reset()
	code = runCommit(&out, &errb, []string{
		"--path", "a.go", "-m", "fix(test): default timeout (fak cmd)",
	})
	if code != 0 {
		t.Fatalf("runCommit exit = %d, stderr = %q", code, errb.String())
	}
	if recordedTimeout != 4*time.Minute {
		t.Fatalf("recordedTimeout = %v, want 4m default", recordedTimeout)
	}
}

func TestCommitBuildCheckTimeoutRefusesWithoutFailingOpen(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go toolchain not on PATH")
	}
	repo, git := seedBuildCheckRepo(t)
	writeBuildCheckFile(t, repo, "go.mod", buildCheckGoMod)
	writeBuildCheckFile(t, repo, "p/p.go", "package p\n\nfunc Value() int { return 1 }\n")
	commitBuildCheckPlumbing(t, repo, git, "seed green", "go.mod", "p/p.go")

	t.Setenv("GOCACHE", t.TempDir())
	headBefore, _ := buildCheckGitState(t, git)

	writeBuildCheckFile(t, repo, "p/p.go", "package p\n\nfunc Value() int { return 2 }\n")

	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--dir", repo,
		"--path", "p/p.go",
		"-m", "fix(p): update value (fak p)",
		"--build-check-timeout", "10ms",
		"--json",
	})
	if code != 3 {
		t.Fatalf("expected exit code 3 (BUILD_CHECK_TIMEOUT); got %d\nstdout=%s\nstderr=%s", code, out.String(), errb.String())
	}

	headAfter, _ := buildCheckGitState(t, git)
	if headBefore != headAfter {
		t.Fatalf("HEAD advanced despite timeout refusal: before=%s after=%s", headBefore, headAfter)
	}

	if !strings.Contains(errb.String(), "BUILD_CHECK_TIMEOUT") {
		t.Fatalf("stderr should headline BUILD_CHECK_TIMEOUT; got:\n%s", errb.String())
	}
	if !strings.Contains(errb.String(), "prospective validation did not finish") {
		t.Fatalf("stderr should explain prospective validation did not finish; got:\n%s", errb.String())
	}

	var res safecommit.Result
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("failed to decode JSON result: %v\noutput=%s", err, out.String())
	}
	if res.Reason != safecommit.ReasonBuildCheckTimeout {
		t.Fatalf("res.Reason = %q, want %q", res.Reason, safecommit.ReasonBuildCheckTimeout)
	}
	if res.Committed {
		t.Fatalf("res.Committed is true, want false")
	}
	if res.BuildCheck == nil {
		t.Fatalf("res.BuildCheck is nil")
	}
	if res.BuildCheck.Outcome != safecommit.BuildCheckSkippedTimeout {
		t.Fatalf("res.BuildCheck.Outcome = %q, want %q", res.BuildCheck.Outcome, safecommit.BuildCheckSkippedTimeout)
	}
	if res.BuildCheck.FailedOpen {
		t.Fatalf("res.BuildCheck.FailedOpen is true, want false (must fail closed)")
	}
}

func TestCommitBuildCheckWithinTimeoutPasses(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go toolchain not on PATH")
	}
	repo, git := seedBuildCheckRepo(t)
	writeBuildCheckFile(t, repo, "go.mod", buildCheckGoMod)
	writeBuildCheckFile(t, repo, "p/p.go", "package p\n\nfunc Value() int { return 1 }\n")
	commitBuildCheckPlumbing(t, repo, git, "seed green", "go.mod", "p/p.go")
	git("checkout", "-B", "main")

	t.Setenv("GOCACHE", t.TempDir())
	headBefore, _ := buildCheckGitState(t, git)

	writeBuildCheckFile(t, repo, "p/p.go", "package p\n\nfunc Value() int { return 2 }\n")

	git("config", "user.email", "t@t")
	git("config", "user.name", "t")
	branch := git("branch", "--show-current")

	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--dir", repo,
		"--trunk", branch,
		"--no-signoff",
		"--path", "p/p.go",
		"-m", "fix(p): update value (fak p)",
		"--build-check-timeout", "2m",
	})
	if code != 0 {
		t.Fatalf("expected exit code 0; got %d\nstdout=%s\nstderr=%s", code, out.String(), errb.String())
	}

	headAfter, _ := buildCheckGitState(t, git)
	if headBefore == headAfter {
		t.Fatalf("HEAD did not advance after successful commit")
	}
}
